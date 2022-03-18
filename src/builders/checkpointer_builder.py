import logging
import torch.optim
from src.core.checkpointers import CustomCheckpointer


def build(checkpoint_dir: str,
          logger: logging.Logger,
          model: dict,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          eval_config: dict,
          phase: str) -> CustomCheckpointer:
    """
    Builds the checkpointer

    :param checkpoint_dir: str, path to save and load checkpoints from
    :param logger: logging.Logger, custom logger
    :param model: dict, dictionary containing submodules
    :param optimizer: torch.optim.Optimizer, torch optimizer
    :param scheduler: torch.optim.lr_scheduler, torch scheduler
    :param eval_config: dict, evaluators config dictionary
    :param phase: str, phase to build the checkpointer for
    :return: the model checkpointer
    """

    checkpointer = CustomCheckpointer(checkpoint_dir=checkpoint_dir,
                                      logger=logger,
                                      model=model,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      eval_config=eval_config,
                                      phase=phase)

    logger.info_important('Checkpointer is built.')

    return checkpointer

