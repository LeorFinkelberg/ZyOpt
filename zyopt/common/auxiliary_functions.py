import time
import typing as t
from functools import wraps

from zyopt.common.logger import make_logger

logger = make_logger(__file__)


def timer(f: t.Callable):
    """
    Measures the execution time of the function
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start_record = time.process_time()
        f(*args, **kwargs)
        logger.info(f"Full calculation time: {(time.process_time() - start_record) / 60:.5f} [min]")

    return wrapper
