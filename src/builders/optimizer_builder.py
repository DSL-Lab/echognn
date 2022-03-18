import logging
import torch.optim
from torch import optim
from copy import deepcopy


OPTIMIZERS = {'adam': optim.Adam}


def build(config: dict,
          model: dict,
          logger: logging.Logger) -> torch.optim.Optimizer:
    """
    Builds the optimizer

    :param config: dict, train config dict
    :param model: dict, dictionary containing submodules
    :param logger: logging.Logger, custom logger
    :return: torch optimizer
    """

    optimizer_config = deepcopy(config)
    optimizer_name = optimizer_config.pop('name')

    # Gather all parameters
    params = list()
    for model_key in model:
        params += list(model[model_key].parameters())

    optimizer = OPTIMIZERS[optimizer_name](params=params,
                                           **optimizer_config)

    logger.info_important('{} optimizer is built.'.format(optimizer_name.upper()))

    return optimizer
