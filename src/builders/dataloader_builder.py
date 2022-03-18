import logging
import torch_geometric.data
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np


def build(config: dict,
          dataset: torch_geometric.data.Dataset,
          logger: logging.Logger,
          phase: str = 'training') -> dict:
    """
    Builds the dataloaders

    :param config: dict, data config dictionary
    :param dataset: torch_geometric.data.Dataset, dataset
    :param logger: logging.Logger, custom logger
    :param phase: str, training or test phase
    :return: dataloaders dictionary
    """

    dataloaders = {}

    if phase == 'pretrain':
        sampler = SubsetRandomSampler(np.arange(start=0, stop=len(dataset['train'])))
        dataloaders['train'] = DataLoader(dataset['train'],
                                          batch_size=config['batch_size'],
                                          drop_last=True,
                                          sampler=sampler,
                                          num_workers=config['num_workers'],
                                          pin_memory=True)

        sampler = SubsetRandomSampler(np.arange(start=0, stop=len(dataset['val'])))
        dataloaders['val'] = DataLoader(dataset['val'],
                                        batch_size=config['batch_size'],
                                        drop_last=True,
                                        sampler=sampler,
                                        num_workers=config['num_workers'],
                                        pin_memory=True)

    # Create samplers for each split
    if phase == 'training':
        train_sampler = SubsetRandomSampler(dataset.train_idx)
        val_sampler = SubsetRandomSampler(dataset.val_idx)

        dataloaders['train'] = DataLoader(dataset,
                                          batch_size=config['batch_size'],
                                          drop_last=True,
                                          sampler=train_sampler,
                                          num_workers=config['num_workers'],
                                          pin_memory=True)

        dataloaders['val'] = DataLoader(dataset,
                                        batch_size=1,
                                        drop_last=True,
                                        sampler=val_sampler,
                                        num_workers=config['num_workers'],
                                        pin_memory=True)

        logger.info_important('Dataloaders are built with {} training '
                              'and {} validation samples.'.format(len(train_sampler),
                                                                  len(val_sampler)))

    elif phase == 'test':
        test_sampler = SubsetRandomSampler(dataset.test_idx)

        dataloaders['test'] = DataLoader(dataset,
                                         batch_size=1,
                                         drop_last=True,
                                         sampler=test_sampler,
                                         num_workers=1)

        logger.info_important('Test Phase Dataloader is built.')
        logger.info_important('Using {} test samples.'.format(len(test_sampler)))

    return dataloaders
