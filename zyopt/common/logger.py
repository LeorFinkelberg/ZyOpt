import logging.config
import sys


def make_stream_handler(
    format_line: str,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    style: str = "{",
) -> logging.StreamHandler:
    """
    Configures stream handler
    """
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(format_line, datefmt=datefmt, style=style))

    return stream_handler


def make_logger(logger_name: str) -> logging.Logger:
    """
    Configures logger
    """
    stream_format = "{asctime}: {levelname} ->> {message}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(make_stream_handler(format_line=stream_format))

    return logger
