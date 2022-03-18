import logging


class AverageEpochMeter(object):
    """
    Loss meter class

    Attributes
    ----------
    name: str, name of the meter
    logger: logging.Logger, custom logger
    sum: float, holds the sum of metric over epochs
    avg: float, holds the average of metric over epochs
    count: int, num of values added to the meter

    Methods
    -------
    reset(): resets avg, sum and count
    update(val): updates the sum and average by adding new value
    """
    def __init__(self, name: str, logger: logging.Logger):
        """
        :param name: str, name of the meter
        :param logger: logging.Logger, custom logger
        """

        self.name = name
        self.logger = logger
        self.reset()
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        """
        resets avg, sum and count
        """

        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        updates the sum and average by adding new value

        :param val: float, value to add
        :param n: int, multiplier
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
