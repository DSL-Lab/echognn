import logging
import os
import torch


class CustomCheckpointer(object):
    """
    Checkpointer class to save trained models

    Attributes
    ----------
    checkpoint_dir: str, Path to save to or load from the checkpoints
    logger: logging.Logger, custom logger
    model: dict, dictionary containing submodules
    optimizer: torch.optim.Optimizer, torch optimizer
    scheduler: torch.optim.lr_scheduler, torch LR scheduler
    eval_standard: str, the evaluator used to compare performance
    minimize: bool, indicates whether the metric is to be maximized or minimized
    best_eval_metric: float, holds the best metric value
    phase: str, the phase the checkpointer is built for

    Methods
    -------
    reset(): reset best eval metric
    save(epoch, eval_metric): saves best model (if performance has improved)
    """

    def __init__(self,
                 checkpoint_dir: str,
                 logger: logging.Logger,
                 model: dict,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 eval_config: dict,
                 phase: str = 'training'):
        """
        :param checkpoint_dir: str, Path to save to or load from the checkpoints
        :param logger: logging.Logger, custom logger
        :param model: dict, dictionary containing submodules
        :param optimizer: torch.optim.Optimizer, torch optimizer
        :param scheduler: torch.optim.lr_scheduler, torch LR scheduler
        :param eval_config: dict, evaluators config dictionary
        :param phase: str, submodules differ based on phase
        """

        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_standard = eval_config['standard']
        self.minimize = eval_config['minimize']
        self.best_eval_metric = 0
        self.phase = phase

        self.reset()

    def reset(self):
        """
        Resets best eval metric
        """
        if self.minimize:
            self.best_eval_metric = 9999999
        else:
            self.best_eval_metric = -9999999

    def save(self, epoch: int, eval_metric: float = None):
        """
        Saves model checkpoint if performance has increased

        :param epoch: int, epoch number
        :param eval_metric: float, the performance evaluator value
        """

        checkpoint = {'epoch': epoch,
                      'eval_standard': self.eval_standard,
                      'best_metric': eval_metric}

        # Add model state dicts
        try:
            checkpoint['video_encoder_model'] = self.model['video_encoder'].module.state_dict()
        except AttributeError:
            checkpoint['video_encoder_model'] = self.model['video_encoder'].state_dict()

        try:
            checkpoint['attention_encoder_model'] = self.model['attention_encoder'].module.state_dict()
        except AttributeError:
            checkpoint['attention_encoder_model'] = self.model['attention_encoder'].state_dict()

        if self.phase != 'pretrain':
            try:
                checkpoint['graph_regressor_model'] = self.model['graph_regressor'].module.state_dict()
            except AttributeError:
                checkpoint['graph_regressor_model'] = self.model['graph_regressor'].state_dict()

        # Add optimizer state dicts
        checkpoint['optimizer'] = self.optimizer.state_dict()

        # Add scheduler state dicts
        checkpoint['scheduler'] = self.scheduler.state_dict()

        # Save last_checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_last.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info('last_checkpoint is saved for epoch {}.'.format(epoch))

        # Save the best_checkpoint if performance improved
        if self.minimize:
            if eval_metric >= self.best_eval_metric:
                return
        else:
            if eval_metric <= self.best_eval_metric:
                return

        # Update best eval metric
        self.best_eval_metric = eval_metric

        # Save best_checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info('best_checkpoint is saved for epoch {} with eval metric {}.'.format(epoch, eval_metric))

    def load(self, checkpoint_path: str = None):
        """
        Loads checkpoint from path

        :param checkpoint_path: str, path to load checkpoint from
        :return: checkpoint with model and optimizer states removed
        """

        checkpoint = None

        if checkpoint_path:
            self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            # Load models
            self.model['video_encoder'].load_state_dict(checkpoint.pop('embedder_model'))
            self.model['attention_encoder'].load_state_dict(checkpoint.pop('encoder_model'))

            if self.phase != 'pretrain':
                self.model['graph_regressor'].load_state_dict(checkpoint.pop('regressor_model'))

            # Load optimizers
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

            # Load scheduler
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

        return checkpoint
