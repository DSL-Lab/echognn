import logging
import torch
from src.core import models
from copy import deepcopy


MODELS = {'video_encoder': models.VideoEncoder,
          'attention_encoder': models.AttentionEncoder,
          'graph_regressor': models.GraphRegressor}


def build(config: dict,
          logger: logging.Logger,
          device: torch.device) -> dict:

    """
    Builds the models dict

    :param config: dict, model config dict
    :param logger: logging.Logger, custom logger
    :param device: torch.device, device to move the models to
    :return: dictionary containing all the submodules (PyTorch models)
    """

    config = deepcopy(config)
    _ = config.pop('checkpoint_path')

    # Create the models
    model = {}
    for model_key in config.keys():
        model[model_key] = MODELS[model_key](config=config[model_key]).to(device)

    logger.info_important('Model is built.')

    return model
