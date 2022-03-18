import logging
from src.core.evaluators import R2Evaluator, MAEEvaluator, F1ScoreEvaluator, RMSEEvaluator, BinaryAccuracyEvaluator

EVALUATORS = {'r2': R2Evaluator,
              'mae': MAEEvaluator,
              'f1score': F1ScoreEvaluator,
              'rmse': RMSEEvaluator,
              'node_binary_accuracy': BinaryAccuracyEvaluator,
              'edge_binary_accuracy': BinaryAccuracyEvaluator}


def build(config: dict,
          logger: logging.Logger) -> dict:
    """
    Builds the dictionary of evaluators

    :param config: dict, evaluators config dictionary
    :param logger: logging.Logger, custom logger
    :return: dictionary containing evaluators
    """

    evaluators = dict()
    for standard in config['standards']:
        evaluators.update({standard: EVALUATORS[standard]()})

        logger.info_important('{} evaluator is built.'.format(standard.upper()))

    return evaluators
