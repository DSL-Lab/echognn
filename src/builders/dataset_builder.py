import logging
from copy import deepcopy
import torch_geometric.data
from src.core.datasets import EchoNetEfDataset, PretrainEchoNetEfDataset


DATASETS = {'echonet': EchoNetEfDataset}
DATASETS_PRETRAIN = {'echonet-pretrain': PretrainEchoNetEfDataset}


def build(config: dict,
          logger: logging.Logger) -> torch_geometric.data.Dataset:
    """
    Builds the dataset

    :param config: dict, data config dictionary
    :param logger: logging.Logger, custom logger
    :return: PyTorch dataset
    """

    data_config = deepcopy(config)
    name = data_config.pop('name')

    if name in DATASETS:
        dataset = DATASETS[name](**data_config)
    elif name in DATASETS_PRETRAIN:
        dataset = dict()
        dataset['train'] = DATASETS_PRETRAIN[name](**data_config, train=True)
        dataset['val'] = DATASETS_PRETRAIN[name](**data_config, train=False)

    logger.info_important('{} dataset is built.'.format(name.upper()))

    return dataset
