import logging
from src.core.meters import AverageEpochMeter


def build(logger: logging.Logger,
          config: dict) -> dict:
    """
    Builds the loss meters

    :param logger: logging.Logger, custom logger
    :param config: dict, criteria config dictionary
    :return: dictionary containing loss meters
    """

    loss_meters = dict()
    for criterion_key in config:
        if criterion_key == 'sparsity':
            loss_meters.update({'node_'+criterion_key: AverageEpochMeter('node ' + criterion_key + ' loss meter',
                                                                         logger)})
            loss_meters.update({'edge_'+criterion_key: AverageEpochMeter('edge ' + criterion_key + ' loss meter',
                                                                         logger)})
        else:
            loss_meters.update({criterion_key: AverageEpochMeter(criterion_key + ' loss meter', logger)})

    logger.info_important('Loss meters are built.')

    return loss_meters
