import os
import shutil
import argparse
from src.engine import Engine
from src.utils import create_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML config file")
    parser.add_argument('--save_dir', default='./logs/run-temp', help='Directory to save config and model checkpoint')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test', action='store_true', default=False,
                       help='Only run testing - ensure the checkpoint path is provided in the config file')
    group.add_argument('--pretrain', action='store_true', default=False,
                       help='Run pretraining - ensure the correct config is provided through the config_path argument')
    args = parser.parse_args()

    # Extract arguments
    save_dir = args.save_dir
    config_path = args.config_path
    test = args.test
    pretrain = args.pretrain

    # Create the save directory
    os.makedirs(save_dir, exist_ok=True)

    # Copy the provided config file into save_dir
    shutil.copyfile(config_path, os.path.join(save_dir, "config.yaml"))

    # Create the logger
    logger = create_logger(name=save_dir)

    # Create the engine (this will create an Engine object that contains models, optimizers, loss functions, etc.)
    engine = Engine(config_path=args.config_path, logger=logger, save_dir=args.save_dir)

    if test:
        engine.evaluate()
    elif pretrain:
        engine.pretrain()
    else:
        engine.train()
