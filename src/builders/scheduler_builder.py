import logging
from copy import deepcopy
import torch
from torch import optim


SCHEDULERS = {'multi-step': optim.lr_scheduler.MultiStepLR}


def build(config: dict,
          optimizer: torch.optim.Optimizer,
          logger: logging.Logger) -> torch.optim.lr_scheduler:
    """
    Builds the LR scheduler

    :param config: dict, train config dictionary
    :param optimizer: torch.optim.Optimizer, torch optimizer
    :param logger: logging.Logger, custom logger
    :return: torch LR scheduler
    """

    schedule_config = deepcopy(config)
    scheduler_name = schedule_config.pop('name')

    # Create all required schedulers
    scheduler = SCHEDULERS[scheduler_name](optimizer=optimizer, **schedule_config)

    logger.info_important('{} scheduler is built.'.format(scheduler_name.upper()))

    return scheduler
