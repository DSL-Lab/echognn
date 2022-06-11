import logging
import os.path
import numpy as np
import yaml
import torch
from src.builders import model_builder, optimizer_builder, scheduler_builder, criteria_builder, meter_builder, \
    evaluator_builder, checkpointer_builder, dataset_builder, dataloader_builder
from src.utils import to_train, to_eval, count_parameters, reset_evaluators, update_evaluators, compute_evaluators, \
    print_epoch_results, reset_meters, update_meters, save_echo_graph, compute_ed_frame_distance, \
    print_es_ed_dist_summary, compute_es_frame_distance, wandb_log
import time
from torch_geometric.utils import to_dense_adj
from src.utils import draw_ef_plots

# Optional Packages
try:
    import wandb
except ImportError:
    print('engine.py: wandb must be installed if use_wandb is set to True!')


class Engine(object):
    """
    Engine class containing all required components for training and evaluating the model

    Attributes
    ----------
    config_path: str, path to config file
    logger: logging.Logger, custom logger
    save_dir: str, path to save checkpoints to
    device: torch.device, device to use
    data_config: dict, dataset config dictionary
    train_config: dict, general training config dictionary
    model_config: dict, model config dictionary
    eval_config: dict, evaluators config dictionary
    apply_shrinkage: bool, indicates whether shrinkage loss is used
    shrinkage_a: float, the "a" parameter of shrinkage loss
    shrinkage_c: float, the "c" parameter of shrinkage loss
    node_sparsity_lambda: float, the weight given to node sparsity loss
    edge_sparsity_lambda: float, the weight given to edge sparsity loss
    classification_lambda: float, the weight given to classification loss
    model: dict, dictionary containing PyTorch models for each submodule in the network
    batch_mask: torch.tensor, mask indicating which nodes belong to the graph for which samples
    sample_weights: torch.tensor, weights used to re-weight loss associated with samples based on their freq in data
    sample_intervals: numpy.ndarray, the intervals for the frequency of samples based on EF values
    dataloader: dict, dictionary containing PyTorch dataloaders for trainin, validation and test set
    optimizer: torch.optim.Optimizer, PyTorch optimizer
    scheduler: torch.optim.lr_scheduler, PyTorch LR scheduler
    criteria: dict, dictionary containing model's objective functions
    loss_meters: src.core.meters.AverageEpochMeter, epoch meter to keep track of model loss
    checkpointer: src.core.CustomCheckpointer, checkpointer to save trained models
    evaluators: dict, dictionary of evaluators to measure model's performance
    misc: dict, extra information extracted from checkpoint

    Methods
    -------
    train(): train the model (training+validation)
    evaluate(): test the model using the provided checkpoint in the config file
    """

    def __init__(self,
                 config_path: str,
                 logger: logging.Logger,
                 save_dir: str):
        """
        :param config_path: str, path to config file
        :param logger: logging.Logger, custom logger
        :param save_dir: str, path to save checkpoints to
        """

        self.logger = logger
        self.save_dir = save_dir

        # Determine the device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.num_devices = torch.cuda.device_count()
        self.logger.warning('Using {} GPU/s!'.format(self.num_devices))

        # Load and process the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config

        # Set up Wandb if required
        if config['train']['use_wandb']:
            wandb.init(project=config['train']['wand_project_name'],
                       config=config,
                       mode=config['train']['wandb_mode'])

    def train(self):
        """
        Start the training process
        """

        self._process_config_file(self.config, phase='training')
        self._build(phase='training')
        self._train()

    def evaluate(self):
        """
        Test the model
        """

        self._process_config_file(self.config, phase='test')
        self._build(phase='test')
        self._evaluate_once(epoch=0, phase='test')

    def pretrain(self):
        """
        Pretrain the model
        """
        self._process_config_file(self.config, phase='pretrain')
        self._build(phase='pretrain')
        self._pretrain()

    def _process_config_file(self, config: dict, phase: str = 'training'):
        """
        Processes the config dict, extracting required attributes based on phase

        :param phase: str, one of training, test or pretrain
        :param config: dict, loaded YAML config file
        """

        # Extract configurations for each component
        self.data_config = config['dataset']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.model_config = config['model']

        # Add configs that the Attention Encoder needs
        self.model_config['attention_encoder'].update({'num_frames': self.data_config['num_frames']})
        self.model_config['attention_encoder'].update({'device': self.device})
        self.model_config['attention_encoder'].update({'input_dim': self.model_config['video_encoder']['output_dim']})

        # train and test exclusive configurations
        if phase == 'pretrain':

            self.node_reweight_ones_by = self.train_config['criteria']['node_location'].pop('reweight_ones_by')
            self.edge_reweight_ones_by = self.train_config['criteria']['edge_location'].pop('reweight_ones_by')

        else:

            # Add sparsity loss type based on whether node or edge distribution is produced by the encoder
            self.train_config['criteria']['sparsity'].update({'encoder_type':
                                                              self.model_config['attention_encoder']['name']})
            self.train_config['criteria']['sparsity'].update({'device': self.device})

            # Extract sparsity loss configs
            self.node_sparsity_lambda = self.train_config['criteria']['sparsity'].pop('node_lambda')
            self.edge_sparsity_lambda = self.train_config['criteria']['sparsity'].pop('edge_lambda')

            # Ensure the loss function doesn't take the average of loss
            self.train_config['criteria']['regression'].update({'reduction': 'none'})

            # Add configs that the Graph Regressor needs
            self.model_config['graph_regressor'].update({'num_frames': self.data_config['num_frames']})
            self.model_config['graph_regressor'].update({'num_clips_per_vid': self.data_config['num_clips_per_vid']})
            self.model_config['graph_regressor'].update({'input_dim': self.model_config['video_encoder']['output_dim']})
            self.model_config['graph_regressor'].update({'num_classes': len(self.data_config['classification_classes'])
                                                                        - 1})

            # Extract regression loss configs
            self.apply_shrinkage = self.train_config['criteria']['regression'].pop('apply_shrinkage')
            self.shrinkage_a = self.train_config['criteria']['regression'].pop('shrinkage_a')
            self.shrinkage_c = self.train_config['criteria']['regression'].pop('shrinkage_c')

            # Extract classification loss configs
            self.classification_lambda = self.train_config['criteria']['classification'].pop('lambda')

    def _build(self,
               phase: str = 'training'):
        """
        Builds the framework's components

        :param phase: str, indicates whether the model is in training or test phase
        """

        # Build the datasets
        dataset = dataset_builder.build(config=self.data_config,
                                        logger=self.logger)

        if phase != 'pretrain':
            # Add number of videos per samples to configs that need it
            self.num_vids_per_sample = dataset.num_vids_per_sample
            self.model_config['attention_encoder'].update({'num_vids_per_sample': dataset.num_vids_per_sample})
            self.model_config['graph_regressor'].update({'num_vids_per_sample': dataset.num_vids_per_sample})

        # Build the model
        self.model = model_builder.build(config=self.model_config, logger=self.logger, device=self.device)

        if phase != 'pretrain':
            # Create the batch mask
            self.batch_mask = torch.tensor(np.repeat(list(range(self.train_config['batch_size'] *
                                                                self.data_config['num_clips_per_vid'])),
                                                     self.data_config['num_frames'] *
                                                     dataset.num_vids_per_sample),
                                           device=self.device,
                                           dtype=torch.long)

            # Get sample weights and associated label intervals needed for weight resampling
            self.sample_weights = None
            self.sample_intervals = None
            if self.train_config['sample_reweighting']:
                self.sample_weights = torch.from_numpy(dataset.sample_weights).float().to(self.device)
                self.sample_intervals = dataset.sample_intervals

        # Build the dataloaders
        self.dataloader = dataloader_builder.build(config=self.train_config,
                                                   dataset=dataset,
                                                   logger=self.logger,
                                                   phase=phase)

        # Build the optimizer
        self.optimizer = optimizer_builder.build(config=self.train_config['optimizer'],
                                                 model=self.model,
                                                 logger=self.logger)

        # Build the scheduler
        self.scheduler = scheduler_builder.build(config=self.train_config['scheduler'],
                                                 optimizer=self.optimizer,
                                                 logger=self.logger)

        # Build the objective functions
        self.criteria = criteria_builder.build(config=self.train_config['criteria'], logger=self.logger)

        # Build the loss meters
        self.loss_meters = meter_builder.build(logger=self.logger, config=self.train_config['criteria'])

        # Build the model evaluators
        self.evaluators = evaluator_builder.build(config=self.eval_config, logger=self.logger)

        # Build the checkpointer
        self.checkpointer = checkpointer_builder.build(checkpoint_dir=self.save_dir,
                                                       logger=self.logger,
                                                       model=self.model,
                                                       optimizer=self.optimizer,
                                                       scheduler=self.scheduler,
                                                       eval_config=self.eval_config,
                                                       phase=phase)
        self.misc = self.checkpointer.load(checkpoint_path=self.model_config.get('checkpoint_path', ''))

        if self.num_devices > 1:
            self.logger.info("Using data parallel for the Video Encoder.")
            self.model['video_encoder'] = torch.nn.DataParallel(self.model['video_encoder'],
                                                                device_ids=list(range(self.num_devices)))

        # Load pretrained models if needed
        if phase != 'pretrain':
            if self.model_config['pretrained_path'] is not None:
                self.model['video_encoder'].load_state_dict(torch.load(os.path.join(self.model_config['pretrained_path'],
                                                                                    'video_encoder.pt')))
                self.model['attention_encoder'].load_state_dict(torch.load(os.path.join(self.model_config['pretrained_path'],
                                                                                        'attention_encoder.pt')))

        # Print number of parameters if needed
        if self.train_config['print_model_stats']:
            count_parameters(self.model['video_encoder'], 'Video Encoder Network', self.logger)
            count_parameters(self.model['attention_encoder'], 'Attention Encoder Network', self.logger)
            if phase == 'training':
                count_parameters(self.model['graph_regressor'], 'Graph Regressor Network', self.logger)

    def _pretrain(self):
        """
        Perform pretraining
        """

        if self.misc:
            start_epoch = self.misc['epoch']
        else:
            start_epoch = 0
        num_epochs = self.train_config.get('num_epochs', 100)
        self.logger.info('Pretrain for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        # Start training iterations
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # reset meters and evaluators
            reset_meters(self.loss_meters)
            reset_evaluators(self.evaluators)

            # Training epoch
            self._pre_train_one_epoch(epoch)

            # reset meters and evaluators
            reset_meters(self.loss_meters)
            reset_evaluators(self.evaluators)

            # Validation epoch
            self._pre_eval_one_epoch(epoch=epoch)

            # Scheduler step
            self.scheduler.step()

    def _train(self):
        """
        Perform training and validation for multiple epochs
        """

        if self.misc:
            start_epoch = self.misc['epoch']
        else:
            start_epoch = 0
        num_epochs = self.train_config.get('num_epochs', 100)
        self.logger.info('Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        # Start training iterations
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # reset meters and evaluators
            reset_meters(self.loss_meters)
            reset_evaluators(self.evaluators)

            # Training epoch
            self._train_one_epoch(epoch)

            # reset meters and evaluators
            reset_meters(self.loss_meters)
            reset_evaluators(self.evaluators)

            # Validation epoch
            self._evaluate_once(epoch=epoch, phase='val')

            # Scheduler step
            self.scheduler.step()

    def _train_one_epoch(self, epoch: int):
        """
        Performs one epoch of training
        :param epoch: int, epoch number
        """
        self.logger.info('Training epoch {} has started.'.format(epoch))

        # Start timer
        train_start = time.time()

        # Move model to training mode
        to_train(self.model)

        # Get the training dataloader
        trainloader = self.dataloader['train']

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for data in trainloader:

            # Move data to correct device
            data = data.to(self.device)

            # Zero the grads
            self.optimizer.zero_grad()

            # Extract different components in the data
            x = data.x
            edge_index = data.edge_index
            regression_labels = data.regression_y
            classification_labels = data.classification_y

            # Create embeddings from video inputs
            x = self.model['video_encoder'](x)

            # Get node and edge weights
            node_weights, edge_weights = self.model['attention_encoder'](x)

            # Create the weighted adjacency matrix for the Graph Regressor
            adj = to_dense_adj(edge_index,
                               edge_attr=torch.flatten(edge_weights[:, :, -1] / torch.max(edge_weights[:, :, -1], 1,
                                                                                          keepdim=True)[0]),
                               batch=self.batch_mask).squeeze(-1)
            adj = adj + torch.eye(adj.shape[-1], device=self.device)

            # Add self loops to the adj matrix
            regression_predictions, classification_predictions = self.model['graph_regressor'](x=x,
                                                                                               frame_weights=
                                                                                               node_weights[:, :, -1],
                                                                                               adj=adj)

            # Find sparsity loss
            node_sparsity_loss, edge_sparsity_loss = self.criteria['sparsity'](node_weights, edge_weights)

            # Compute regression loss
            regression_loss = self.criteria['regression'](regression_predictions, regression_labels)

            # Apply loss shrinkage if needed
            if self.apply_shrinkage:
                shrinkage_factor = torch.sigmoid(self.shrinkage_a *
                                                 (self.shrinkage_c -
                                                  torch.abs(regression_predictions - regression_labels)))
                regression_loss = shrinkage_factor * regression_loss

            # Compute classification loss
            classification_loss = self.criteria['classification'](classification_predictions, classification_labels)

            # Perform sample weighting if needed
            if self.sample_weights is not None:
                sample_weights = np.digitize(regression_labels.detach().cpu().numpy(), self.sample_intervals) - 1
                sample_weights = self.sample_weights[sample_weights]
                regression_loss = regression_loss * sample_weights

            # Average loss over samples
            regression_loss = regression_loss.mean()

            # Compute total loss
            loss = regression_loss + \
                   self.classification_lambda * classification_loss + \
                   self.node_sparsity_lambda * node_sparsity_loss + self.edge_sparsity_lambda * edge_sparsity_loss

            # Backprop
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                # Update loss meters
                update_meters(self.loss_meters, {'regression': regression_loss.detach().item(),
                                                 'classification': classification_loss.detach().item(),
                                                 'node_sparsity': node_sparsity_loss.detach().item(),
                                                 'edge_sparsity': edge_sparsity_loss.detach().item()})

                try:
                    update_evaluators(self.evaluators,
                                      regression_predictions.detach().cpu().numpy()[:, 0],
                                      regression_labels.detach().cpu().numpy())
                except IndexError:
                    update_evaluators(self.evaluators,
                                      regression_predictions.detach().cpu().numpy(),
                                      regression_labels.detach().cpu().numpy())

        # Compute training time
        train_time = time.time() - train_start
        with torch.no_grad():

            # Compute the evaluation metrics
            eval_metrics = compute_evaluators(self.evaluators)

            # Compute total loss
            total_loss = self.loss_meters['regression'].avg + \
                         self.classification_lambda * self.loss_meters['classification'].avg + \
                         self.node_sparsity_lambda * self.loss_meters['node_sparsity'].avg + \
                         self.edge_sparsity_lambda * self.loss_meters['edge_sparsity'].avg

            # Print epoch summary
            print_epoch_results(logger=self.logger,
                                phase='Training',
                                epoch=epoch,
                                elapsed_time=train_time,
                                total_loss=total_loss,
                                losses={'regression': self.loss_meters['regression'].avg,
                                        'classification': self.loss_meters['classification'].avg,
                                        'node sparsity': self.loss_meters['node_sparsity'].avg,
                                        'edge sparsity': self.loss_meters['edge_sparsity'].avg},
                                eval_metrics=eval_metrics)

            if self.train_config['use_wandb']:
                wandb_log(phase='training',
                          epoch=epoch,
                          losses={'regression': self.loss_meters['regression'].avg,
                                  'classification': self.loss_meters['classification'].avg,
                                  'node_sparsity': self.loss_meters['node_sparsity'].avg,
                                  'edge_sparsity': self.loss_meters['edge_sparsity'].avg,
                                  'total_loss': total_loss},
                          eval_metrics=eval_metrics)

    def _evaluate_once(self, epoch: int, phase: str = 'val'):
        """
        Performs one epoch of evaluation

        :param epoch: int, epoch number
        :param phase: str, validation or test phase
        """

        self.logger.info('Evaluation epoch {} has started.'.format(epoch))
        eval_start = time.time()

        # Print the validation metric for the checkpoint
        if phase == 'test':
            self.logger.info_important('Validation metric: {}'.format(self.misc['best_metric']))

        to_eval(self.model)

        # Summary dicts for ES and ED
        ed_summary_dict = {'num_invalid_labels': 0,
                           'dist': [],
                           'num_failures': 0,
                           'not_determinable': 0,
                           'unaccounted_corner_case': 0,
                           'num_all_ones': 0,
                           'num_all_zeros': 0}

        es_summary_dict = {'num_invalid_labels': 0,
                           'dist': [],
                           'num_failures': 0,
                           'not_determinable': 0,
                           'unaccounted_corner_case': 0,
                           'num_all_ones': 0,
                           'num_all_zeros': 0}

        with torch.no_grad():

            # Get dataloader
            evalloader = self.dataloader[phase]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Hold all ytrue and ypreds to draw the EF scatter plot
            if self.train_config['eval_visualization']:
                ytrue = np.array([])
                ypred = np.array([])

            for data in evalloader:
                # Move data to correct device
                data = data.to(self.device)

                # Extract different components in the data
                x = data.x
                edge_index = data.edge_index
                regression_labels = data.regression_y
                classification_labels = data.classification_y

                # Batch mask looks different during test time
                eval_batch_mask = torch.tensor(np.repeat(list(range(x.shape[0])),
                                                         self.data_config['num_frames']*self.num_vids_per_sample),
                                               device=self.device,
                                               dtype=torch.long)

                # Create embeddings from video inputs
                x = self.model['video_encoder'](x)

                # Get node and edge weights
                node_weights, edge_weights = self.model['attention_encoder'](x)

                # Create the weighted adjacency matrix for the Graph Regressor
                adj = to_dense_adj(edge_index,
                                   edge_attr=torch.flatten(edge_weights[:, :, -1] / torch.max(edge_weights[:, :, -1], 1,
                                                                                              keepdim=True)[0]),
                                   batch=eval_batch_mask).squeeze(-1)
                adj = adj + torch.eye(adj.shape[-1], device=self.device)

                # Add self loops to the adj matrix
                regression_predictions, classification_predictions = self.model['graph_regressor'](x=x,
                                                                                                   frame_weights=
                                                                                                   node_weights[:, :,
                                                                                                                -1],
                                                                                                   adj=adj,
                                                                                                   phase=phase)

                # Add to array of ground truth labels and predictions
                if self.train_config['eval_visualization']:
                    ytrue = np.concatenate((ytrue, regression_labels.detach().cpu().numpy()),
                                           axis=0) if ytrue.size else regression_labels.detach().cpu().numpy()
                    ypred = np.concatenate((ypred, regression_predictions.detach().cpu().numpy()),
                                           axis=0) if ypred.size else regression_predictions.detach().cpu().numpy()

                # Find sparsity loss
                node_sparsity_loss, edge_sparsity_loss = self.criteria['sparsity'](node_weights, edge_weights)

                # Compute regression loss
                regression_loss = self.criteria['regression'](regression_predictions, regression_labels).mean()

                # Compute classification loss
                classification_loss = self.criteria['classification'](classification_predictions, classification_labels)

                # Update loss meters
                update_meters(self.loss_meters, {'regression': regression_loss.detach().item(),
                                                 'classification': classification_loss.detach().item(),
                                                 'node_sparsity': node_sparsity_loss.detach().item(),
                                                 'edge_sparsity': edge_sparsity_loss.detach().item()})

                try:
                    update_evaluators(self.evaluators,
                                      regression_predictions.detach().cpu().numpy()[:, 0],
                                      regression_labels.detach().cpu().numpy())
                except IndexError:
                    update_evaluators(self.evaluators,
                                      regression_predictions.detach().cpu().numpy(),
                                      regression_labels.detach().cpu().numpy())

                # Create visualization if needed
                if self.train_config['eval_visualization']:
                    if data.x.shape[0] == 1:
                        frame_idx = data.frame_idx
                    else:
                        frame_idx = data.frame_idx[0]

                    for i in range(data.x.shape[0]):
                        save_echo_graph(echo_clip=data.x[i, 0].detach().cpu().numpy(),
                                        node_weights=node_weights[i, :, -1].detach().cpu().numpy(),
                                        edge_weights=adj[i].detach().cpu().numpy(),
                                        es_frame_idx=data.es_frame[0].detach().cpu().numpy(),
                                        ed_frame_idx=data.ed_frame[0].detach().cpu().numpy(),
                                        all_frame_idx=frame_idx[i],
                                        experiment_name='r2_'+str(self.misc['best_metric']),
                                        clip_num=i,
                                        save_path='./visualizations',
                                        loss=regression_loss[0].detach().cpu().item())

                # Compute ES/ED frames distances
                compute_ed_frame_distance(ed_frame_true=data.ed_frame[0].detach().cpu().item(),
                                          es_frame_true=data.es_frame[0].detach().cpu().item(),
                                          summary_dict=ed_summary_dict,
                                          threshold=30,
                                          num_ones_to_reject=55,
                                          num_zeros_to_reject=60,
                                          frame_idx=data.frame_idx if data.x.shape[0] == 1 else data.frame_idx[0],
                                          num_frames=self.data_config['num_frames'],
                                          weights_to_use='outgoing_edge',
                                          adj=adj,
                                          frame_weights=node_weights)

                compute_es_frame_distance(ed_frame_true=data.ed_frame[0].detach().cpu().item(),
                                          es_frame_true=data.es_frame[0].detach().cpu().item(),
                                          summary_dict=es_summary_dict,
                                          threshold=15,
                                          num_ones_to_reject=55,
                                          num_zeros_to_reject=60,
                                          frame_idx=data.frame_idx if data.x.shape[0] == 1 else data.frame_idx[0],
                                          num_frames=self.data_config['num_frames'],
                                          weights_to_use='outgoing_edge',
                                          adj=adj,
                                          frame_weights=node_weights)

            if self.train_config['eval_visualization']:
                draw_ef_plots(predictions=ypred,
                              labels=ytrue,
                              experiment_name='r2_'+str(self.misc['best_metric']),
                              path='./visualizations')

            # Compute test time
            eval_time = time.time() - eval_start

            # Compute the evaluation metrics
            eval_metrics = compute_evaluators(self.evaluators)

            # Save model if performance improved
            if phase == 'val':
                self.checkpointer.save(epoch, eval_metrics[self.eval_config['standard']])

            # Compute total loss
            total_loss = self.loss_meters['regression'].avg + \
                         self.classification_lambda * self.loss_meters['classification'].avg + \
                         self.node_sparsity_lambda * self.loss_meters['node_sparsity'].avg + \
                         self.edge_sparsity_lambda * self.loss_meters['edge_sparsity'].avg

            # Print epoch summary
            print_epoch_results(logger=self.logger,
                                phase='Evaluation',
                                epoch=epoch,
                                elapsed_time=eval_time,
                                total_loss=total_loss,
                                losses={'regression': self.loss_meters['regression'].avg,
                                        'classification': self.loss_meters['classification'].avg,
                                        'node sparsity': self.loss_meters['node_sparsity'].avg,
                                        'edge sparsity': self.loss_meters['edge_sparsity'].avg},
                                eval_metrics=eval_metrics)

            if self.train_config['use_wandb']:
                wandb_log(phase=phase,
                          epoch=epoch,
                          losses={'regression': self.loss_meters['regression'].avg,
                                  'classification': self.loss_meters['classification'].avg,
                                  'node_sparsity': self.loss_meters['node_sparsity'].avg,
                                  'edge_sparsity': self.loss_meters['edge_sparsity'].avg,
                                  'total_loss': total_loss},
                          eval_metrics=eval_metrics)

            # Print ES/ED dist summary
            print_es_ed_dist_summary(ed_summary_dict=ed_summary_dict,
                                     es_summary_dict=es_summary_dict,
                                     logger=self.logger)

    def _pre_train_one_epoch(self, epoch: int):
        """
        Performs one epoch of pretraining
        :param epoch: int, epoch number
        """
        self.logger.info('Pretraining epoch {} has started.'.format(epoch))

        # Start timer
        train_start = time.time()

        # Move model to training mode
        to_train(self.model)

        # Get the training dataloader
        trainloader = self.dataloader['train']

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for data in trainloader:

            # Move data to correct device
            data = data.to(self.device)

            # Zero the grads
            self.optimizer.zero_grad()

            # Extract different components in the data
            x, node_y, edge_y, node_mask, \
            edge_mask, edge_index = data.x, data.node_y, data.edge_y, data.node_mask, data.edge_mask, data.edge_index

            # Create embeddings from video inputs
            x = self.model['video_encoder'](x)

            # Get node and edge weights
            node_weights, edge_weights = self.model['attention_encoder'](x)

            # Compute loss
            node_loss = self.criteria['node_location'](torch.flatten(node_weights), node_y) * node_mask
            edge_loss = self.criteria['edge_location'](torch.flatten(edge_weights), edge_y) * edge_mask

            # Weight samples
            if self.node_reweight_ones_by > 1:
                node_sample_weights = node_y.detach().cpu().numpy()
                node_sample_weights[node_sample_weights == 1] = self.node_reweight_ones_by
                node_sample_weights[node_sample_weights == 0] = 1
                node_sample_weights = torch.from_numpy(node_sample_weights).float().to(self.device)
                node_loss = node_loss * node_sample_weights

            if self.edge_reweight_ones_by > 1:
                edge_sample_weights = edge_y.detach().cpu().numpy()
                edge_sample_weights[edge_sample_weights == 1] = self.edge_reweight_ones_by
                edge_sample_weights[edge_sample_weights == 0] = 1
                edge_sample_weights = torch.from_numpy(edge_sample_weights).float().to(self.device)
                edge_loss = edge_loss * edge_sample_weights

            node_loss = torch.mean(node_loss, dim=0)
            edge_loss = torch.mean(edge_loss, dim=0)
            loss = node_loss + edge_loss

            # Backprop
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                # Update loss meters
                update_meters(self.loss_meters, {'node_location': node_loss.detach().item(),
                                                 'edge_location': edge_loss.detach().item()})

                # Update the evaluators
                self.evaluators['node_binary_accuracy'].update(torch.flatten(node_weights).detach().cpu().numpy(),
                                                               node_y.detach().cpu().numpy())
                self.evaluators['edge_binary_accuracy'].update(torch.flatten(edge_weights).detach().cpu().numpy(),
                                                               edge_y.detach().cpu().numpy())

        # Compute training time
        train_time = time.time() - train_start
        with torch.no_grad():

            # Compute the evaluation metrics
            eval_metrics = compute_evaluators(self.evaluators)

            # Compute total loss
            total_loss = self.loss_meters['node_location'].avg + self.loss_meters['edge_location'].avg

            # Print epoch summary
            print_epoch_results(logger=self.logger,
                                phase='Training',
                                epoch=epoch,
                                elapsed_time=train_time,
                                total_loss=total_loss,
                                losses={'node location': self.loss_meters['node_location'].avg,
                                        'edge location': self.loss_meters['edge_location'].avg},
                                eval_metrics=eval_metrics)

            if self.train_config['use_wandb']:
                wandb_log(phase='training',
                          epoch=epoch,
                          losses={'node_location': self.loss_meters['node_location'].avg,
                                  'edge_location': self.loss_meters['edge_location'].avg,
                                  'total_loss': total_loss},
                          eval_metrics=eval_metrics)

    def _pre_eval_one_epoch(self, epoch: int):
        """
        Performs one epoch of validation in the pretraining phase

        :param epoch: int, epoch number
        """

        self.logger.info('Evaluation epoch {} has started.'.format(epoch))
        eval_start = time.time()

        to_eval(self.model)

        with torch.no_grad():

            # Get dataloader
            evalloader = self.dataloader['val']

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for data in evalloader:
                # Move data to correct device
                data = data.to(self.device)

                # Extract different components in the data
                x, node_y, edge_y, node_mask, edge_mask, \
                edge_index = data.x, data.node_y, data.edge_y, data.node_mask, data.edge_mask, data.edge_index

                # Create embeddings from video inputs
                x = self.model['video_encoder'](x)

                # Get node and edge weights
                node_weights, edge_weights = self.model['attention_encoder'](x)

                # Compute loss
                node_loss = self.criteria['node_location'](torch.flatten(node_weights), node_y) * node_mask
                edge_loss = self.criteria['edge_location'](torch.flatten(edge_weights), edge_y) * edge_mask
                node_loss = torch.mean(node_loss, dim=0)
                edge_loss = torch.mean(edge_loss, dim=0)
                loss = node_loss + edge_loss

                # Update loss meters
                update_meters(self.loss_meters, {'node_location': node_loss.detach().item(),
                                                 'edge_location': edge_loss.detach().item()})

                # Update the evaluators
                self.evaluators['node_binary_accuracy'].update(torch.flatten(node_weights).detach().cpu().numpy(),
                                                               node_y.detach().cpu().numpy())
                self.evaluators['edge_binary_accuracy'].update(torch.flatten(edge_weights).detach().cpu().numpy(),
                                                               edge_y.detach().cpu().numpy())

            # Compute test time
            eval_time = time.time() - eval_start

            # Compute the evaluation metrics
            eval_metrics = compute_evaluators(self.evaluators)

            # Save model if performance improved
            total_loss = self.loss_meters['node_location'].avg + self.loss_meters['edge_location'].avg
            self.checkpointer.save(epoch, total_loss)

            # Print epoch summary
            print_epoch_results(logger=self.logger,
                                phase='Validation',
                                epoch=epoch,
                                elapsed_time=eval_time,
                                total_loss=total_loss,
                                losses={'node location': self.loss_meters['node_location'].avg,
                                        'edge location': self.loss_meters['edge_location'].avg},
                                eval_metrics=eval_metrics)

            if self.train_config['use_wandb']:
                wandb_log(phase='val',
                          epoch=epoch,
                          losses={'node_location': self.loss_meters['node_location'].avg,
                                  'edge_location': self.loss_meters['edge_location'].avg,
                                  'total_loss': total_loss},
                          eval_metrics=eval_metrics)
