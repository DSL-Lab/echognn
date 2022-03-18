import os
from abc import ABC
import numpy
import torch_geometric.data
from torch_geometric.data import Dataset
import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.nn import Upsample
import cv2
from random import randint
from math import isnan


class EchoNetEfDataset(Dataset, ABC):
    """
    Dataset class for EchoNet-Dynamic dataset
    The dataset can be found at: https://echonet.github.io/dynamic/

    Attributes
    ----------
    num_frames: int, number of frames per clip
    num_vids_per_sample: int, number of videos per patient
    num_clips_per_vid: int, number of clips per video
    num_frames_per_cycle: int, approximate number of frames per cardiac cycle
    train_idx: numpy.ndarray, list indices indicating CSV rows belonging to the train set
    val_idx: numpy.ndarray, list indices indicating CSV rows belonging to the validation set
    test_idx: numpy.ndarray, list indices indicating CSV rows belonging to the test set
    es_frames: torch.tensor, tensor containing ES frame indices
    ed_frames: torch.tensor, tensor containing ED frame indices
    patient_data_dirs: list['str'], list containing the directory to each sample video
    regression_labels: np.ndarray, numpy array containing regression labels
    classification_labels: np.ndarray, numpy array containing classification labels
    sample_weights: np.ndarray, numpy array containing weights for each sample based on EF frequency
    sample_intervals: np.ndarray, the bins for EF used to find sample_weights
    num_samples: int, number of samples in the dataset
    trans: torchvision.transforms.Compose, torchvision transformation for each data sample
    zoom_aug: bool, indicates whether LV zoom-in augmentation is used during training
    upsample: torch.nn.Upsample, performs 2D upsampling (needed for zomm augmentation)
    test_clip_overlap: int, number of frames to overlap for test clips (if None, no overlap except for the last clip)
    """

    def __init__(self,
                 dataset_path: str,
                 num_frames: int = 32,
                 num_clips_per_vid: int = 1,
                 mean: float = 0.1289,
                 std: float = 0.1911,
                 label_string: str = 'EF',
                 label_div: float = 1.0,
                 num_frames_per_cycle: int = 64,
                 classification_classes: numpy.ndarray = None,
                 zoom_aug: bool = False,
                 test_clip_overlap: int = 0):
        """
        :param dataset_path: str, path to dataset directory
        :param num_frames: int, number of frames per clip
        :param num_clips_per_vid: int, number of clips per video
        :param mean: float, mean used in data standardization
        :param std: float, std used in data standardization
        :param label_string: str, string indicating which column in dataset CSV is for the labels
        :param label_div: float, value to divide labels by (for example, EF can be normalized between 0-1)
        :param num_frames_per_cycle: int, approximate number of frames per cardiac cycle
        :param classification_classes: numpy.ndarray, the intervals for classification classes
        :param zoom_aug: bool, indicates whether LV zoom-in augmentation is used during training
        :param test_clip_overlap: int, number of frames to overlap for each test clip (if 0, no overlap except for
                                  the last clip)
        """

        super().__init__()

        # Default classification classes
        if classification_classes is None:
            classification_classes = np.array([0, 30, 40, 55, 100])

        # CSV file containing file names and labels
        filelist_df = pd.read_csv(os.path.join(dataset_path, 'FileList.csv'))

        # Extract Split information
        splits = np.array(filelist_df['Split'].tolist())
        self.train_idx = np.where(splits == 'TRAIN')[0]
        self.val_idx = np.where(splits == 'VAL')[0]
        self.test_idx = np.where(splits == 'TEST')[0]

        # Extract ES and ED frame indices
        self.es_frames = torch.tensor(np.array(filelist_df['ESFrame']), dtype=torch.int32)
        self.ed_frames = torch.tensor(np.array(filelist_df['EDFrame']), dtype=torch.int32)

        # Extract video file names
        filenames = np.array(filelist_df['FileName'].tolist())

        # All file paths
        self.patient_data_dirs = [os.path.join(dataset_path,
                                               'Videos',
                                               file_name + '.avi')
                                  for file_name
                                  in filenames.tolist()]

        # Get the EF labels for regression and classification
        self.regression_labels = np.array(filelist_df[label_string].tolist())
        self.classification_labels = np.digitize(self.regression_labels, classification_classes) - 1
        self.classification_labels = torch.tensor(self.classification_labels, dtype=torch.long)
        self.regression_labels = torch.tensor(self.regression_labels / label_div,
                                              dtype=torch.float32)

        # Create sample weights based on histogram of sample frequency
        hist, bins = np.histogram(self.regression_labels, bins=60)
        hist = hist + 100
        hist = np.clip(hist, a_min=0, a_max=400)
        hist = 1/hist / np.max(1/hist)
        self.sample_weights = hist
        self.sample_intervals = bins
        self.sample_intervals[0] = 0
        self.sample_intervals[-1] = 1.0

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((mean), (std))])

        # Interpolation needed if augmentation is required
        self.upsample = None
        if zoom_aug:
            self.upsample = Upsample(size=(112, 112), mode='nearest')

        # Other attributes
        self.num_frames = num_frames
        self.num_vids_per_sample = 1
        self.num_clips_per_vid = num_clips_per_vid
        self.num_frames_per_cycle = num_frames_per_cycle
        self.zoom_aug = zoom_aug
        self.test_clip_overlap = test_clip_overlap

    def __getitem__(self, idx: int) -> torch_geometric.data.data:
        """
        Fetch a sample from the dataset

        :param idx: int, index to extract from the dataset
        :return: torch_geometric.data.data, PyG data
        """

        # Get the label
        regression_label = self.regression_labels[idx]
        classification_label = self.classification_labels[idx]

        # Get the video
        cine_vid = self._loadvideo(self.patient_data_dirs[idx])

        # Transform video
        cine_vid = self.trans(cine_vid)

        # Perform augmentation during training
        if (idx in self.train_idx) and self.zoom_aug:
            if np.random.randint(0, 2):
                # Hardcoded for now
                cine_vid = cine_vid[:,  0:90, 20:92].unsqueeze(1)
                cine_vid = self.upsample(cine_vid).squeeze(1)

        # Test behaviour and Train behaviour are different
        if idx in self.test_idx or idx in self.val_idx:
            cine_vid, frame_idx = self.extract_test_data(cine_vid)
        else:
            cine_vid, frame_idx = self.extract_train_data(cine_vid)

        # Interpolate if needed
        if cine_vid.shape[2] < self.num_frames:
            cine_vid = torch.cat((cine_vid, torch.zeros(cine_vid.shape[0], 1,
                                                        self.num_frames - cine_vid.shape[2], 112, 112)),
                                 dim=2)

        # Create fully connected graph
        nx_graph = nx.complete_graph(self.num_frames, create_using=nx.DiGraph())
        for i in range(1, cine_vid.shape[0]):
            nx_graph = nx.compose(nx_graph, nx.complete_graph(range(i*self.num_frames,
                                                                    (i+1)*self.num_frames),
                                                              create_using=nx.DiGraph()))

        g = from_networkx(nx_graph)

        # Add images and label to graph
        g.x = cine_vid
        g.regression_y = regression_label
        g.classification_y = classification_label
        g.es_frame = self.es_frames[idx]
        g.ed_frame = self.ed_frames[idx]
        g.vid_dir = self.patient_data_dirs[idx]
        g.frame_idx = frame_idx

        return g

    def extract_test_data(self, cine_vid: torch.tensor) -> (torch.tensor, np.ndarray):
        """
        Extract the test data

        :param cine_vid: torch.tensor, input tensor of shape T*H*W
        :return: return the extracted cine video (torch tensor) of shape num_clips*1*num_frames*H*W
        """

        # Extract number of frames per video
        video_num_frames = cine_vid.shape[0]

        # Extract the initial frame for each clip
        initial_frames = np.arange(start=0, stop=video_num_frames, step=self.num_frames - self.test_clip_overlap)

        # Extract the first clip
        try:
            cine_vids = cine_vid[0: self.num_frames].unsqueeze(0)
            frame_idx = np.arange(start=0, stop=self.num_frames)
        except (IndexError, RuntimeError) as e:
            cine_vids = cine_vid[0:].unsqueeze(0)
            frame_idx = np.arange(start=0, stop=video_num_frames)

        # Extract consecutive clips
        if self.num_frames < video_num_frames:
            # Get back to back clips
            for initial_idx in initial_frames[1:]:
                try:
                    cine_vids = torch.cat([cine_vids,
                                           cine_vid[initial_idx: initial_idx + self.num_frames].unsqueeze(0)], dim=0)
                    frame_idx = np.vstack([frame_idx, np.arange(start=initial_idx, stop=initial_idx + self.num_frames)])

                # If the last clip overshoots the video, overlap it with the previous clip
                except (IndexError, RuntimeError) as e:
                    cine_vids = torch.cat([cine_vids,
                                           cine_vid[video_num_frames - self.num_frames:].unsqueeze(0)], dim=0)
                    frame_idx = np.vstack([frame_idx, np.arange(start=video_num_frames - self.num_frames,
                                                                stop=video_num_frames)])

        cine_vid = cine_vids.unsqueeze(1)
        return cine_vid, frame_idx

    def extract_train_data(self, cine_vid: torch.tensor) -> (torch.tensor, np.ndarray):
        """
        Extract the train data

        :param cine_vid: torch.tensor, input tensor of shape T*H*W
        :return: return the extracted cine video (torch tensor) of shape num_clips_per_vid*1*num_frames*H*W
        """

        # Extract number of frames per video
        video_num_frames = cine_vid.shape[0]

        # if the required number of frames is larger than the available number of frames in the video
        # take the whole video
        if self.num_frames > video_num_frames:
            cine_vid = cine_vid[list(range(0, video_num_frames))].unsqueeze(0).unsqueeze(1)
            frame_idx = np.arange(start=0, stop=video_num_frames)
            # Use the same video multiple times since not enough frames are available
            if self.num_clips_per_vid > 1:
                cine_vid = torch.tensor(np.vstack([cine_vid] * self.num_clips_per_vid))
                frame_idx = np.vstack([frame_idx] * self.num_clips_per_vid)

        # take num_frames from the whole range of the video if number of frame per cycle is larger than video length
        elif self.num_frames_per_cycle > video_num_frames:
            frame_idx = np.floor(np.linspace(0,
                                             video_num_frames - 1,
                                             self.num_frames)).astype(np.int32).tolist()
            cine_vid = cine_vid[frame_idx].unsqueeze(0).unsqueeze(1)
            if self.num_clips_per_vid > 1:
                cine_vid = torch.tensor(np.vstack([cine_vid] * self.num_clips_per_vid))
                frame_idx = np.vstack([frame_idx] * self.num_clips_per_vid)

        # Follow the procedure outlined in the paper
        else:
            initial_frame = randint(0, video_num_frames - self.num_frames_per_cycle)
            frame_idx_prim = np.floor(np.linspace(initial_frame,
                                                  initial_frame + self.num_frames_per_cycle - 1,
                                                  self.num_frames)).astype(np.int32).tolist()
            cine_vid_prim = cine_vid[frame_idx_prim].unsqueeze(0).unsqueeze(1)

            for i in range(1, self.num_clips_per_vid):
                initial_frame = randint(0, video_num_frames - self.num_frames_per_cycle)
                frame_idx_temp = np.floor(np.linspace(initial_frame,
                                                      initial_frame + self.num_frames_per_cycle - 1,
                                                      self.num_frames)).astype(np.int32).tolist()
                cine_vid_temp = cine_vid[frame_idx_temp].unsqueeze(0).unsqueeze(1)
                cine_vid_prim = torch.tensor(np.vstack((cine_vid_prim, cine_vid_temp)))
                frame_idx_prim = np.vstack((frame_idx_prim, frame_idx_temp))

            cine_vid = cine_vid_prim
            frame_idx = frame_idx_prim

        return cine_vid, frame_idx

    def __len__(self):
        """
        Returns number of samples in the dataset

        :return: number of samples in the dataset
        """

        return self.num_samples

    @staticmethod
    def _loadvideo(filename: str):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v


class PretrainEchoNetEfDataset(Dataset, ABC):
    """
    Dataset class for EchoNet-Dynamic dataset used by the pretraining script
    The dataset can be found at: https://echonet.github.io/dynamic/

    Attributes
    ----------
    num_frames: int, number of frames per clip
    es_frames: torch.tensor, tensor containing ES frame indices
    ed_frames: torch.tensor, tensor containing ED frame indices
    patient_data_dirs: list['str'], list containing the directory to each sample video
    num_samples: int, number of samples in the dataset
    trans: torchvision.transforms.Compose, torchvision transformation for each data sample
    zoom_aug: bool, indicates whether LV zoom-in augmentation is used during training
    upsample: torch.nn.Upsample, performs 2D upsampling (needed for zoom augmentation)
    spread_label_by: int, number of frames around ES/ED frames to label as ES/ED
    num_neighbourhood_frames: int, number of frames around ES/ED frames to consider (other don't affect loss)
    train: bool, indicates whether the dataset is created for training
    """

    def __init__(self,
                 dataset_path: str,
                 num_frames: int = 64,
                 mean: float = 0.1289,
                 std: float = 0.1911,
                 train: bool = True,
                 num_neighbourhood_frames: int = 10,
                 spread_label_by: int = 1,
                 zoom_aug: bool = False,
                 valid_es_ed_dist_thresh: int = 10):
        """
        :param dataset_path: str, path to dataset directory
        :param num_frames: int, number of frames per clip
        :param mean: float, mean used in data standardization
        :param std: float, std used in data standardization
        :param train: bool, indicates whether the training or validation set are to be used
        :param num_neighbourhood_frames: int, number of frames around ES/ED frames to consider (other don't affect loss)
        :param spread_label_by: int, number of frames around ES/ED frames to label as ES/ED
        :param zoom_aug: bool, indicates whether LV zoom-in augmentation is used during training
        :param valid_es_ed_dist_thresh: int, if the distance between ES and ED is smaller than this, don't include it
        """

        super().__init__()

        # CSV file containing file names and labels
        filelist_df = pd.read_csv(os.path.join(dataset_path, 'FileList.csv'))

        # Extract Split information
        splits = np.array(filelist_df['Split'].tolist())
        if train:
            split_idx = np.where(splits == 'TRAIN')[0]
        else:
            split_idx = np.where(splits == 'VAL')[0]

        # Find the distance between ES and ED frames
        es_frames = np.array(filelist_df['ESFrame'])[split_idx]
        ed_frames = np.array(filelist_df['EDFrame'])[split_idx]

        # Find distance between ES and ED frames
        distances_bw_es_ed = es_frames - ed_frames
        distances_bw_es_ed = [-100 if isnan(distance_bw_es_ed) else
                              int(distance_bw_es_ed) for
                              distance_bw_es_ed in distances_bw_es_ed]

        # Extract video file names
        filenames = np.array(filelist_df['FileName'].tolist())

        # All file paths for the required split
        self.patient_data_dirs = np.array([os.path.join(dataset_path,
                                               'Videos',
                                               file_name + '.avi')
                                           for file_name
                                           in filenames.tolist()])
        self.patient_data_dirs = self.patient_data_dirs[split_idx]

        # Remove files that have bad ES and ED frame indices
        self.patient_data_dirs = [patient_data_dir for i, patient_data_dir in enumerate(self.patient_data_dirs)
                                  if (distances_bw_es_ed[i] > valid_es_ed_dist_thresh)]

        # Extract ES/ED frame locations
        self.es_frames = [int(es_frame) for i, es_frame in enumerate(es_frames)
                          if (distances_bw_es_ed[i] > valid_es_ed_dist_thresh)]
        self.ed_frames = [int(ed_frame) for i, ed_frame in enumerate(ed_frames)
                          if (distances_bw_es_ed[i] > valid_es_ed_dist_thresh)]

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((mean), (std))])

        # Interpolation needed if augmentation is required
        self.upsample = None
        if zoom_aug:
            self.upsample = Upsample(size=(112, 112), mode='nearest')

        # Other attributes
        self.num_frames = num_frames
        self.num_neighbourhood_frames = num_neighbourhood_frames
        self.spread_label_by = spread_label_by
        self.zoom_aug = zoom_aug
        self.train = train

    def __getitem__(self, idx: int) -> torch_geometric.data.data:
        """
        Fetch a sample from the dataset

        :param idx: int, index to extract from the dataset
        :return: torch_geometric.data.data, PyG data
        """

        frame_focus = np.random.randint(0, 2)

        if frame_focus == 0:
            primary_foi = self.es_frames[idx]
            secondary_foi = self.ed_frames[idx]
        else:
            primary_foi = self.ed_frames[idx]
            secondary_foi = self.es_frames[idx]

        # Get the video
        cine_vid = self._loadvideo(self.patient_data_dirs[idx])
        # Transform video
        cine_vid = self.trans(cine_vid)

        # Perform augmentation during training
        if self.train and self.zoom_aug:
            if np.random.randint(0, 2):
                # Hardcoded for now
                cine_vid = cine_vid[:,  0:90, 20:92].unsqueeze(1)
                cine_vid = self.upsample(cine_vid).squeeze(1)

        # Get number of frames in the video
        video_num_frames = cine_vid.shape[0]

        # Labels and mask
        node_mask = self.set_mask(video_num_frames, primary_foi, secondary_foi)
        node_label = self.set_label(video_num_frames, primary_foi, secondary_foi)

        # Extract the required frames
        if self.num_frames > video_num_frames:
            frame_idx = np.arange(start=0, stop=video_num_frames)
        else:
            initial_frame_start = max(0, primary_foi - self.num_frames)
            initial_frame = randint(initial_frame_start, primary_foi+1)
            last_frame = min(initial_frame+self.num_frames, video_num_frames)
            frame_idx = np.arange(initial_frame, last_frame)

        # Extract correct label, mask and video frames
        cine_vid = cine_vid[frame_idx].unsqueeze(0).unsqueeze(1)
        node_label = node_label[frame_idx]
        node_mask = node_mask[frame_idx]

        # Add labels for 0-frames that will be added if video is too short
        if len(node_label) < self.num_frames:
            node_label = np.concatenate((node_label, np.zeros(self.num_frames - len(node_label))))
        if len(node_mask) < self.num_frames:
            node_mask = np.concatenate((node_mask, np.zeros(self.num_frames - len(node_mask))))

        # Interpolate if needed
        if cine_vid.shape[2] < self.num_frames:
            cine_vid = torch.cat((cine_vid, torch.zeros(1, 1, self.num_frames-cine_vid.shape[2], 112, 112)), dim=2)

        # Create fully connected graph
        g = from_networkx(nx.complete_graph(cine_vid.shape[2], create_using=nx.DiGraph()))

        # Create edge labels (outgoing and incoming edges of ES/ED frames)
        edge_index = g.edge_index.detach().cpu().numpy()
        edge_label = np.zeros((self.num_frames * (self.num_frames - 1)))
        noi = np.where(node_label == 1)[0]
        for node_idx in noi:
            for i in range(2):
                noi_edge_idx = np.where(edge_index[i] == node_idx)[0]
                edge_label[noi_edge_idx] = 1

        # Create edge mask (edges that affect the loss function)
        edge_mask = np.zeros((self.num_frames * (self.num_frames - 1)))
        noi = np.where(node_mask == 1)[0]
        for node_idx in noi:
            for i in range(2):
                noi_edge_idx = np.where(edge_index[i] == node_idx)[0]
                edge_mask[noi_edge_idx] = 1

        # Add images and label to graph
        g.x = cine_vid
        g.node_y = torch.tensor(node_label, dtype=torch.float32)
        g.edge_y = torch.tensor(edge_label, dtype=torch.float32)
        g.node_mask = torch.tensor(node_mask, dtype=torch.long)
        g.edge_mask = torch.tensor(edge_mask, dtype=torch.long)

        return g

    def __len__(self):
        """
        Returns number of samples in the dataset

        :return: number of samples in the dataset
        """

        return self.num_samples

    @staticmethod
    def _loadvideo(filename):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v

    def set_mask(self,
                 video_num_frames: int,
                 primary_foi: int,
                 secondary_foi: int) -> numpy.ndarray:
        """
        Create the node mask such that only nodes close to ES/ED frames affect the loss function

        :param video_num_frames: int, number of frames in the video
        :param primary_foi: int, index of primary frame (clip will surely include this frame)
        :param secondary_foi: int, index of secondary frame
        :return: numpy array of node masks
        """

        mask = np.zeros(video_num_frames)

        mask[max(primary_foi-self.num_neighbourhood_frames, 0):
             min(primary_foi+self.num_neighbourhood_frames + 1, video_num_frames)] = 1
        mask[max(secondary_foi-self.num_neighbourhood_frames, 0):
             min(secondary_foi+self.num_neighbourhood_frames + 1, video_num_frames)] = 1

        return mask

    def set_label(self,
                  video_num_frames: int,
                  primary_foi: int,
                  secondary_foi: int) -> numpy.ndarray:
        """
        Create the node labels such that ES/ED frames and spread_label_by frames around them are set to 1

        :param video_num_frames: int, number of frames in the video
        :param primary_foi: int, index of primary frame (clip will surely include this frame)
        :param secondary_foi: int, index of secondary frame
        :return: numpy array of node labels
        """

        label = np.zeros(video_num_frames)

        label[max(primary_foi-self.spread_label_by, 0):
              min(primary_foi+self.spread_label_by + 1, video_num_frames)] = 1
        label[max(secondary_foi-self.spread_label_by, 0):
              min(secondary_foi+self.spread_label_by + 1, video_num_frames)] = 1

        return label
