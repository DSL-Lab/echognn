import numpy
import torch.nn
from colorlog import ColoredFormatter
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os

# Optional Packages
try:
    from prettytable import PrettyTable
except ImportError:
    print('prettytable must be installed if print_model_stats is set to True!')


try:
    import wandb
except ImportError:
    print('wandb must be installed if use_wandb is set to True!')


def create_logger(name: str) -> logging.Logger:
    """
    Creates a custom logger

    :param name: str, name for the logger
    :return: A custom logging.logger object
    """

    # Define a new level for the logger
    def _info_important(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOIMPORTANT':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.propagate = False
    logger.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOIMPORTANT')
    logging.Logger.info_important = _info_important

    return logger


def to_train(model: dict):
    """
    Calls the train() method for each model in the dictionary

    :param model: dict, dictionary containing PyTroch nn.Module s
    """

    for model_name in model.keys():
        model[model_name].train()


def to_eval(model: dict):
    """
    Calls the eval() method for each model in the dictionary

    :param model: dict, dictionary containing PyTroch nn.Module s
    """

    for model_name in model.keys():
        model[model_name].eval()


def reset_evaluators(evaluators):
    """
    Calls the reset() method of evaluators in input dict

    :param evaluators: dict, dictionary of evaluators
    """

    for evaluator in evaluators.keys():
        evaluators[evaluator].reset()


def update_evaluators(evaluators, y_pred, y_true):
    """
    Calls the update() method of evaluators in input dict to add y_pred and y_true

    :param evaluators: dict, dictionary of evaluators
    :param y_pred: numpy.ndarray, array of predictions
    :param y_true: numpy.ndarray, array of ground truth labels
    """

    for evaluator in evaluators.keys():
        evaluators[evaluator].update(y_pred, y_true)


def compute_evaluators(evaluators):
    """
    Calls the compute() method of evaluatros in input dict

    :param evaluators: dict, dictionary of evaluators
    :return: evaluated metrics
    """

    eval_metrics = dict()
    for evaluator in evaluators:
        eval_metrics.update({evaluator: evaluators[evaluator].compute()})

    return eval_metrics


def reset_meters(meters: dict):
    """
    Calls the reset() method of meters in input dict

    :param evaluators: dict, dictionary of meters
    """

    for meter in meters.keys():
        meters[meter].reset()


def update_meters(meters: dict, values: dict):
    """
    Update the meters with given values

    :param meters: dict, dictionary of meters
    :param values: dict, dictionary of values to update meters with
    """

    for meter in meters.keys():
        meters[meter].update(values[meter])


def count_parameters(model: torch.nn.Module, model_name: str, logger: logging.Logger) -> int:
    """
    Counts the number of learnable parameters in the model
    This code is copied and modified from https://newbedev.com/check-the-total-number-of-parameters-in-a-pytorch-model

    :param model: torch.nn.Module, PyTorch model to find the parameters for
    :param model_name: str, name of the model
    :param logger: logging.Logger, custom logger
    :return: total number of parameters
    """

    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info('Total Trainable Params for {}: {}'.format(model_name, total_params))

    return total_params


def save_echo_graph(echo_clip: numpy.ndarray,
                    node_weights: numpy.ndarray,
                    edge_weights: numpy.ndarray,
                    es_frame_idx: int,
                    ed_frame_idx: int,
                    all_frame_idx: numpy.ndarray,
                    clip_num: int,
                    save_path: str,
                    experiment_name: str,
                    loss: float = None):
    """

    :param echo_clip: numpy.ndarray, echo frames of shape T*W*H
    :param node_weights: numpy.ndarray, array containing node weights of shape T*1
    :param edge_weights: numpy.ndarray, array containing edge weights of shape T*T
    :param es_frame_idx: int, index indicating which frame is the labelled ES
    :param ed_frame_idx: int, index indicating which frame is the labelled ED
    :param all_frame_idx: numpy.ndarray, list indicating the frame indices of the clip within original echo video
    :param clip_num: int, the clip number for sample
    :param save_path: str, path to save visualization to
    :param experiment_name: str, sub directory in /save_path
    :param loss: int, save visualization to /save_path/experiment_name/loss
    """

    # Create directory to save visualizations to
    path = os.path.join(save_path, experiment_name, str(loss))
    os.makedirs(path, exist_ok=True)

    # Save each frame in the clip
    for i, (weight, img, frame) in enumerate(zip(node_weights, echo_clip, all_frame_idx)):

        # Change numpy array into PIL image
        image = (((img - img.min()) / (img.max() - img.min())) * 255.9).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize((400, 400))

        # Save frame and change name if the frame is labelled as ES/ED
        if es_frame_idx == frame:
            image.save(os.path.join(path,
                                    str(clip_num) + '_frame_' + str(i) + '_weight_' + str(weight) + '_cine_es.png'))
        elif ed_frame_idx == frame:
            image.save(os.path.join(path,
                                    str(clip_num) + '_frame_' + str(i) + '_weight_' + str(weight) + '_cine_ed.png'))
        else:
            image.save(os.path.join(path,
                                    str(clip_num) + '_frame_' + str(i) + '_weight_' + str(weight) + '_cine.png'))

    # Find ES/ED frame locations in the clip
    es_idx = np.where(all_frame_idx == es_frame_idx)[0]
    ed_idx = np.where(all_frame_idx == ed_frame_idx)[0]

    # Create node weights bar plot
    save_echo_graph_bar_plot(title='frame_weights',
                             xlabel='Frame',
                             ylabel='Weight',
                             weights=node_weights,
                             es_idx=es_idx,
                             ed_idx=ed_idx,
                             path=path,
                             clip_num=clip_num,
                             loss=loss)

    # Create incoming edge weights bar plot
    in_edge_weights = np.sum(edge_weights, axis=0)
    save_echo_graph_bar_plot(title='incoming_edge_weights',
                             xlabel='Frame',
                             ylabel='Weight',
                             weights=in_edge_weights,
                             es_idx=es_idx,
                             ed_idx=ed_idx,
                             path=path,
                             clip_num=clip_num,
                             loss=loss)

    out_edge_weights = np.sum(edge_weights, axis=1)
    save_echo_graph_bar_plot(title='outgoing_edge_weights',
                             xlabel='Frame',
                             ylabel='Weight',
                             weights=in_edge_weights,
                             es_idx=es_idx,
                             ed_idx=ed_idx,
                             path=path,
                             clip_num=clip_num,
                             loss=loss)

    # Create and save the adjacency matrix heatmaps
    save_echo_graph_heatmap(title='adj',
                            weights=edge_weights,
                            clip_num=clip_num,
                            path=path)

    save_echo_graph_heatmap(title='incoming',
                            weights=np.tril(edge_weights) + np.tril(edge_weights).T - np.eye(edge_weights[0].shape[0]),
                            clip_num=clip_num,
                            path=path)

    save_echo_graph_heatmap(title='outgoing',
                            weights=np.triu(edge_weights) + np.triu(edge_weights).T - np.eye(edge_weights[0].shape[0]),
                            clip_num=clip_num,
                            path=path)


def save_echo_graph_heatmap(title: str,
                            weights: numpy.ndarray,
                            clip_num: int,
                            path: str):
    """
    Cretes and saves echo-graph weight heatmaps

    :param title: str, title of plot
    :param weights: numpy.ndarray, weights to visualize
    :param clip_num: int, clip number
    :param path: str, path to save the figure to
    """

    svm = sn.heatmap(weights)
    figure = svm.get_figure()
    figure.savefig(os.path.join(path, str(clip_num) + '_' + title + '_heatmap.png'))
    plt.clf()


def save_echo_graph_bar_plot(title: str,
                             xlabel: str,
                             ylabel: str,
                             weights: numpy.ndarray,
                             es_idx: list,
                             ed_idx: list,
                             path: str,
                             clip_num: int,
                             loss: float = None):
    """
    Creates and save bar plot for the echo graph

    :param title: str, title of plot
    :param xlabel: str, x axis title
    :param ylabel: str, y axis title
    :param weights: numpy.ndarray, array of learned weights
    :param es_idx: list, list of es index
    :param ed_idx: list, list of ed index
    :param path: str, path to save the figure to
    :param clip_num: int, clip number
    :param loss: float, loss associated with sample
    """

    fig = plt.figure()
    plt.title(title)
    plt.ylim((0, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    barlist = plt.bar(np.linspace(0,
                                  weights.shape[0],
                                  weights.shape[0]),
                      height=weights, width=0.5)

    # Change the color of bars corresponding to ES/ED
    if len(es_idx) > 0:
        barlist[es_idx[0]].set_color('r')
    if len(ed_idx) > 0:
        barlist[ed_idx[0]].set_color('b')

    # Save figure
    if loss is not None:
        fig.savefig(os.path.join(path, str(clip_num) + '_loss_'+str(loss) + '_' + title + '_histogram.jpg'))
    else:
        fig.savefig(os.path.join(path, str(clip_num) + '_' + title + '_histogram.jpg'))

    plt.clf()


def print_epoch_results(logger: logging.Logger,
                        phase: str,
                        epoch: int,
                        elapsed_time: float,
                        total_loss: float,
                        losses: dict,
                        eval_metrics: dict):
    """
    Prints results for an epoch

    :param logger: logging.Logger, custom logger
    :param phase: str, epoch phase (e.g. training or validation)
    :param epoch: int, epoch number
    :param elapsed_time: float, computation time
    :param total_loss: float, model's total loss
    :param losses: dict, dictionary containing different losses used for the model
    :param eval_metrics: dict, dictionary containing evaluated metrics
    """

    logger.info_important(phase+' Epoch {} - Computation Time {} - Total loss {}'.format(epoch,
                                                                                         elapsed_time,
                                                                                         total_loss))

    for loss_key in losses:
        logger.info_important(loss_key.upper() + ' Loss: {}'.format(losses[loss_key]))

    for eval_metric in eval_metrics:
        logger.info_important(eval_metric.upper()+' Metric: {}'.format(eval_metrics[eval_metric]))


def wandb_log(phase: str,
              epoch: int,
              losses: dict,
              eval_metrics: dict,
              es_summary_dict: dict = None,
              ed_summary_dict: dict = None):
    """
    Log epoch information to Wandb

    :param phase: str, indicates the run phase
    :param epoch: int, epoch number
    :param losses: dict, dictionary of losses
    :param eval_metrics: dict, dictionary containing evaluator results
    :param es_summary_dict: dict, dictionary containing ES frame distance summary
    :param ed_summary_dict: dict, dictionary containing ED frame distance summary
    """

    for loss_key in losses:
        wandb.log({phase+'_{}_loss'.format(loss_key): losses[loss_key]},
                  step=epoch)

    for eval_metric in eval_metrics:
        wandb.log({phase+'_{}_metric'.format(eval_metric): eval_metrics[eval_metric]},
                  step=epoch)

    if es_summary_dict is not None and ed_summary_dict is not None:
        wandb.log({phase + '_ed_dist_len': len(ed_summary_dict['dist']),
                   phase + '_es_dist_len': len(es_summary_dict['dist']),
                   phase + '_ed_dist': np.mean(ed_summary_dict['dist']) if len(ed_summary_dict['dist']) > 0 else 0,
                   phase + '_es_dist': np.mean(es_summary_dict['dist']) if len(es_summary_dict['dist']) > 0 else 0,
                   phase + '_ed_unaccounted': ed_summary_dict['unaccounted_corner_case'],
                   phase + '_es_unaccounted': es_summary_dict['unaccounted_corner_case'],
                   phase + '_ed_not_determinable': ed_summary_dict['not_determinable'],
                   phase + '_es_not_determinable': es_summary_dict['not_determinable'],
                   phase + '_ed_num_all_ones': ed_summary_dict['num_all_ones'],
                   phase + '_es_num_all_ones': es_summary_dict['num_all_ones'],
                   phase + '_ed_num_all_zeros': ed_summary_dict['num_all_zeros'],
                   phase + '_es_num_all_zeros': es_summary_dict['num_all_zeros']},
                  step=epoch)


def draw_ef_plots(predictions: numpy.ndarray,
                  labels: numpy.ndarray,
                  experiment_name: str,
                  path: str = None,
                  figure_num: int = 0,
                  label: str = None):
    """
    Draws the scatter plot and confusion matrix for EF

    :param predictions: numpy.ndarray, predicted EF values
    :param labels: numpy.ndarray, ground truth EF values
    :param experiment_name: str, experiment name to save visualizations to path/experiment_name
    :param path: str, path to save visualization to
    :param figure_num: int, figure number
    :param label: str, extra label to add to plot title
    """

    path = os.path.join(path, experiment_name)
    os.makedirs(path, exist_ok=True)

    fig, ax = plt.subplots()

    ax.scatter(labels, predictions)
    ax.set_xlabel('Ground Truth EF')
    ax.set_ylabel('EF Estimate')

    if label:
        ax.set_title('EF Estimate vs. Ground Truth - ' + label)
    else:
        ax.set_title('EF Estimate vs. Ground Truth')

    # Plot correct True EF line
    # (from: https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if path:
        plt.savefig(os.path.join(path, 'ef_'+str(figure_num)+'.png'))
    else:
        plt.show()

    plt.close()

    class_labels = np.digitize(labels, np.array([0, 0.30, 0.40, 0.55, 1.0])) - 1
    class_preds = np.digitize(predictions, np.array([0, 0.30, 0.40, 0.55, 1.0])) - 1
    conf_mat = confusion_matrix(class_labels, class_preds)
    conf_mat = conf_mat / np.sum(conf_mat, axis=0)
    sn.heatmap(conf_mat, annot=True, xticklabels=['<30%', '30%<=EF<40%', '40%<=EF<55%', '55%<='],
               yticklabels=['<30%', '30%<=EF<40%', '40%<=EF<55%', '55%<='])
    plt.savefig(os.path.join(path, 'confusion_mat_'+str(figure_num)+'.png'))

    plt.close()


def compute_ed_frame_distance(ed_frame_true: int,
                              es_frame_true: int,
                              summary_dict: dict,
                              threshold: float,
                              num_ones_to_reject: int,
                              num_zeros_to_reject: int,
                              frame_idx: list,
                              num_frames: int = 64,
                              weights_to_use: str = 'outgoing_edge',
                              adj: torch.tensor = None,
                              frame_weights: torch.tensor = None):
    """
    Compute the average ED frame distance and other summary metrics

    :param ed_frame_true: int, index for the ground truth ED
    :param es_frame_true: int, index for the ground truth ES
    :param summary_dict: dict, dictionary containing the summary of metrics
    :param threshold: float, the threshold to use for binarizing the weights
    :param num_ones_to_reject: int, reject if the number of 1's in the binary weights is above this
    :param num_zeros_to_reject: int, reject if the number of 0's in the binary weights is above this
    :param frame_idx: list, list of frame indices for each clip
    :param num_frames: int, number of frames in each clip
    :param weights_to_use: str, indicates the weights to use for ED/ES location approximation
    :param adj: torch.tensor, the adjacency matrix of shape N*num_frames*num_frames containing graph weights
    :param frame_weights: torch.tensor, node weights of shape 1*num_frames
    """

    assert weights_to_use in ['incoming_edge', 'outgoing_edge', 'node'], "weights_to_use must be one of" \
                                                                                  "[incoming_edge, outgoing_edge, node]"

    num_clips = len(frame_idx)

    # Check that frame labels are valid
    if ed_frame_true < 0 or es_frame_true < 0:
        summary_dict['num_invalid_labels'] += 1
        return

    # Find the clip number where the ED label is located
    clip_idx = torch.div(ed_frame_true, 64, rounding_mode='floor')

    # Get the frame indices for the clip that contains the ED frame
    clip_frame_idx = frame_idx[clip_idx]

    # Find the frame index of labelled ED in the chosen clip
    clip_ed_idx = np.where(clip_frame_idx == ed_frame_true)[0][0]

    binary_weights = get_binary_weights(summary_dict=summary_dict,
                                        weights_to_use=weights_to_use,
                                        clip_idx=clip_idx,
                                        threshold=threshold,
                                        num_ones_to_reject=num_ones_to_reject,
                                        num_zeros_to_reject=num_zeros_to_reject,
                                        adj=adj,
                                        frame_weights=frame_weights)
    if binary_weights is None:
        return

    # If ES frame index is within the same clip and ahead of ED and all weights between them are 0, the model has failed
    # to capture the cycle
    if es_frame_true in clip_frame_idx and es_frame_true > ed_frame_true:
        clip_es_idx = np.where(clip_frame_idx == es_frame_true)[0][0]
        if np.count_nonzero(binary_weights[clip_ed_idx+1: clip_es_idx] == 1) == 0:
            summary_dict['num_failures'] += 1
            return

    # For the case where ED's weight is below the binary threshold, we should keep looking ahead in the weights until
    # we reach a 1. The distance between that 1 and the labelled ED is the frame distance
    if binary_weights[clip_ed_idx] == 0:
        try:
            approx_frame_dist = np.where(binary_weights[clip_ed_idx:] == 1)[0][0]
            summary_dict['dist'].append(approx_frame_dist)
            return

        except IndexError:

            # If no 1's are found in the current clip, we must check the next clips for the first 1
            # last clip is ignored since it may have an overlap with its previous clip
            next_clip_idx = clip_idx + 1
            if next_clip_idx >= num_clips - 1:
                summary_dict['not_determinable'] += 1
                return

            binary_weights = get_binary_weights(summary_dict=summary_dict,
                                                weights_to_use=weights_to_use,
                                                clip_idx=next_clip_idx,
                                                threshold=threshold,
                                                num_ones_to_reject=num_ones_to_reject,
                                                num_zeros_to_reject=num_zeros_to_reject,
                                                adj=adj,
                                                frame_weights=frame_weights)
            if binary_weights is None:
                return

            # If there are no 1's before the ES frame in the next clip, the model has failed to capture the cycle
            if es_frame_true in frame_idx[next_clip_idx]:
                clip_es_idx = np.where(frame_idx[next_clip_idx] == es_frame_true)[0][0]
                if np.count_nonzero(binary_weights[0: clip_es_idx] == 1) == 0:
                    summary_dict['num_failures'] += 1
                    return

            try:
                approx_frame_dist = np.where(binary_weights[0:] == 1)[0][0]
                summary_dict['dist'].append(num_frames - clip_ed_idx + approx_frame_dist)
                return

            except IndexError:
                summary_dict['unaccounted_corner_case'] += 1
                return

    # For the case where ED's weight is above the binary threshold, we should keep looking back in the weights until
    # we reach a 0. The distance between that 0 and the true ED is the frame distance
    else:
        try:
            approx_frame_dist = abs(clip_ed_idx - np.where(binary_weights[0:clip_ed_idx] == 0)[0][-1] - 1)
            summary_dict['dist'].append(approx_frame_dist)
            return

        except IndexError:

            if binary_weights[clip_ed_idx - 1] == 0:
                summary_dict['dist'].append(1)
                return

            # If no 0's are found in the current clip, we must check the previous clips for the first 0
            prev_clip_idx = clip_idx - 1
            if prev_clip_idx < 0:
                summary_dict['not_determinable'] += 1
                return

            binary_weights = get_binary_weights(summary_dict=summary_dict,
                                                weights_to_use=weights_to_use,
                                                clip_idx=prev_clip_idx,
                                                threshold=threshold,
                                                num_ones_to_reject=num_ones_to_reject,
                                                num_zeros_to_reject=num_zeros_to_reject,
                                                adj=adj,
                                                frame_weights=frame_weights)
            if binary_weights is None:
                return

            try:
                approx_frame_dist = np.where(binary_weights == 0)[0][-1]
                summary_dict['dist'].append(clip_ed_idx + num_frames - approx_frame_dist)
                return

            except IndexError:
                summary_dict['unaccounted_corner_case'] += 1
                return


def compute_es_frame_distance(ed_frame_true: int,
                              es_frame_true: int,
                              summary_dict: dict,
                              threshold: float,
                              num_ones_to_reject: int,
                              num_zeros_to_reject: int,
                              frame_idx: list,
                              num_frames: int = 64,
                              weights_to_use: str = 'outgoing_edge',
                              adj: torch.tensor = None,
                              frame_weights: torch.tensor = None):

    assert weights_to_use in ['incoming_edge', 'outgoing_edge', 'node'], "weights_to_use must be one of" \
                                                                         "[incoming_edge, outgoing_edge, node]"

    num_clips = len(frame_idx)

    # Check that frame labels are valid
    if ed_frame_true < 0 or es_frame_true < 0:
        summary_dict['num_invalid_labels'] += 1
        return

    # Find the clip number where the ED label is located
    clip_idx = torch.div(es_frame_true, 64, rounding_mode='floor')

    # Get the frame indices for the clip that contains the ED frame
    clip_frame_idx = frame_idx[clip_idx]

    # Find the frame index of labelled ED in the chosen clip
    clip_es_idx = np.where(clip_frame_idx == es_frame_true)[0][0]

    binary_weights = get_binary_weights(summary_dict=summary_dict,
                                        weights_to_use=weights_to_use,
                                        clip_idx=clip_idx,
                                        threshold=threshold,
                                        num_ones_to_reject=num_ones_to_reject,
                                        num_zeros_to_reject=num_zeros_to_reject,
                                        adj=adj,
                                        frame_weights=frame_weights)
    if binary_weights is None:
        return

    # If ED frame index is within the same clip and behind ES and all weights between them are 0, the model has failed
    # to capture the cycle
    if ed_frame_true in clip_frame_idx and es_frame_true > ed_frame_true:
        clip_ed_idx = np.where(clip_frame_idx == ed_frame_true)[0][0]
        if np.count_nonzero(binary_weights[clip_ed_idx+1: clip_es_idx] == 1) == 0:
            summary_dict['num_failures'] += 1
            return

    # For the case where ES's weight is below the binary threshold, we should keep looking back in the weights until
    # we reach a 1. The distance between that 1 and the labelled ES is the frame distance
    if binary_weights[clip_es_idx] == 0:
        try:
            approx_frame_dist = abs(clip_es_idx - np.where(binary_weights[0:clip_es_idx] == 1)[0][-1])
            summary_dict['dist'].append(approx_frame_dist)

        except IndexError:

            # If no 1's are found in the current clip, we must check the previous clips for the first 1
            prev_clip_idx = clip_idx - 1
            if prev_clip_idx < 0:
                summary_dict['not_determinable'] += 1
                return

            binary_weights = get_binary_weights(summary_dict=summary_dict,
                                                weights_to_use=weights_to_use,
                                                clip_idx=prev_clip_idx,
                                                threshold=threshold,
                                                num_ones_to_reject=num_ones_to_reject,
                                                num_zeros_to_reject=num_zeros_to_reject,
                                                adj=adj,
                                                frame_weights=frame_weights)
            if binary_weights is None:
                return

            # If there are no 1's after the ED frame in the previous clip, the model has failed to capture the cycle
            if ed_frame_true in frame_idx[prev_clip_idx]:
                clip_ed_idx = np.where(frame_idx[prev_clip_idx] == ed_frame_true)[0][0]
                if np.count_nonzero(binary_weights[clip_ed_idx+1:] == 1) == 0:
                    summary_dict['num_failures'] += 1
                    return

            try:
                approx_frame_dist = np.where(binary_weights[0:] == 1)[0][-1]
                summary_dict['dist'].append(num_frames - approx_frame_dist + clip_es_idx)
                return

            except IndexError:
                summary_dict['unaccounted_corner_case'] += 1
                return

    # For the case where ES's weight is above the binary threshold, we should keep looking ahead in the weights until
    # we reach a 0. The distance between that 0 and the true ES is the frame distance
    else:
        try:
            approx_frame_dist = np.where(binary_weights[clip_es_idx:] == 0)[0][0]
            summary_dict['dist'].append(approx_frame_dist)
            return

        except IndexError:

            # If no 0's are found in the current clip, we must check the next clip for the first 0
            # last clip is ignored since it may have an overlap with its previous clip
            next_clip_idx = clip_idx + 1
            if next_clip_idx >= num_clips - 1:
                summary_dict['not_determinable'] += 1
                return

            binary_weights = get_binary_weights(summary_dict=summary_dict,
                                                weights_to_use=weights_to_use,
                                                clip_idx=next_clip_idx,
                                                threshold=threshold,
                                                num_ones_to_reject=num_ones_to_reject,
                                                num_zeros_to_reject=num_zeros_to_reject,
                                                adj=adj,
                                                frame_weights=frame_weights)
            if binary_weights is None:
                return

            try:
                approx_frame_dist = np.where(binary_weights[0:] == 0)[0][0]
                summary_dict['dist'].append(num_frames - clip_es_idx + approx_frame_dist)
                return

            except IndexError:
                summary_dict['unaccounted_corner_case'] += 1
                return


def get_binary_weights(summary_dict: dict,
                       weights_to_use: str,
                       clip_idx: torch.tensor,
                       threshold: float,
                       num_ones_to_reject: int,
                       num_zeros_to_reject: int,
                       adj: torch.tensor = None,
                       frame_weights: torch.tensor = None):
    """
    Get the binary weights for using the given threshold

    :param summary_dict: dict, summary dictionary of ES or ED
    :param weights_to_use: str, indicates the type of weight to use. Must be one of outgoing_edge, incoming_edge or node
    :param clip_idx: int, index of clip to use
    :param threshold: float, the threshold to use for binarizing the weights
    :param num_ones_to_reject: int, if the number of ones in the binary weights is above this, reject it
    :param num_zeros_to_reject: int, if the number of zeros in the binary weights is above this, reject it
    :param adj: torch.tensor, the adjacency matrix of shapre N*num_frames*num_frames
    :param frame_weights: torch.tensor, node weights of shape N*num_frames
    :return: Binary weight tensor of shape 1*num_frames
    """

    # Fetch the weights to use
    if weights_to_use == 'outgoing_edge':
        weights = np.sum(adj[clip_idx].detach().cpu().numpy(), axis=1)
    elif weights_to_use == 'incoming_edge':
        weights = np.sum(adj[clip_idx].detach().cpu().numpy(), axis=0)
    else:
        weights = frame_weights[clip_idx]

    # Binarize the frame weights based on given threshold
    binary_weights = (weights > threshold).astype(int)

    # Check if number of ones is above the allowed threshold
    if np.sum(binary_weights) >= num_ones_to_reject:
        summary_dict['num_all_ones'] += 1
        return None

    # Check if number of zeros is above the allowed threshold
    if np.count_nonzero(binary_weights == 0) >= num_zeros_to_reject:
        summary_dict['num_all_zeros'] += 1
        return None

    return binary_weights


def print_es_ed_dist_summary(ed_summary_dict: dict,
                             es_summary_dict: dict,
                             logger: logging.Logger):
    """
    Prints the summary for ES/ED calculations

    :param ed_summary_dict: dict, dictionary containing ED summary variables
    :param es_summary_dict: dict, dictionary containing ES summary variables
    :param logger: logging.Logger, custom logger
    """

    logger.info("LEN ED DIST: {}".format(len(ed_summary_dict['dist'])))
    logger.info("LEN ES DIST: {}".format(len(es_summary_dict['dist'])))
    logger.info("ED DIST: {}".format(np.mean(ed_summary_dict['dist']) if len(ed_summary_dict['dist']) > 0 else 0))
    logger.info("ES DIST: {}".format(np.mean(es_summary_dict['dist']) if len(es_summary_dict['dist']) > 0 else 0))
    logger.info("ED Invalid labels: {}".format(ed_summary_dict['num_invalid_labels']))
    logger.info("ES Invalid labels: {}".format(es_summary_dict['num_invalid_labels']))
    logger.info("ED Number of Unaccounted Corner"
                " Cases: {}".format(ed_summary_dict['unaccounted_corner_case']))
    logger.info("ES Number of Unaccounted Corner "
                "Cases: {}".format(es_summary_dict['unaccounted_corner_case']))
    logger.info("ED Non-Determinable: {}".format(ed_summary_dict['not_determinable']))
    logger.info("ES Non-Determinable: {}".format(es_summary_dict['not_determinable']))
    logger.info("ED Failures: {}".format(ed_summary_dict['num_failures']))
    logger.info("ES Failures: {}".format(es_summary_dict['num_failures']))
    logger.info("ED Num All Ones: {}".format(ed_summary_dict['num_all_ones']))
    logger.info("ES Num All Ones: {}".format(es_summary_dict['num_all_ones']))
    logger.info("ED Num All Zeros: {}".format(ed_summary_dict['num_all_zeros']))
    logger.info("ES Num All Zeros: {}".format(es_summary_dict['num_all_zeros']))
