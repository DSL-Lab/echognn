import logging
import torch.nn as nn
from copy import deepcopy
from src.core.criteria import L1SparsityLoss


CRITERIA = {'mae': nn.L1Loss,
            'l1sparsity': L1SparsityLoss,
            'smoothmae': nn.SmoothL1Loss,
            'mse': nn.MSELoss,
            'crossentropy': nn.CrossEntropyLoss,
            'bce': nn.BCELoss}


def build(config: dict, logger: logging.Logger) -> dict:
    """
    Builds the dictionary of criteria

    :param config: dict, criteria config dictionary
    :return: dictionary containing the objective functions
    """
    config = deepcopy(config)

    criteria = dict()
    for criterion_key in config.keys():
        criterion_name = config[criterion_key].pop('name')
        criteria.update({criterion_key: CRITERIA[criterion_name](**config[criterion_key])})

        logger.info_important('{} criterion: {} is built.'.format(criterion_key.upper(), criterion_name.upper()))

    return criteria
