import logging
import sys

LOG_FORMAT = "format=%(asctime)s loglevel=%(levelname)-6s logger=%(name)s %(funcName)s() L%(lineno)-4d %(message)s"
DETAILED_LOG_FORMAT = "format=%(asctime)s loglevel=%(levelname)-6s logger=%(name)s %(funcName)s() L%(lineno)-4d %(message)s   call_trace=%(pathname)s L%(lineno)-4d"  # noqa


def get_logger(name: str) -> logging.Logger:
    this_logger = logging.getLogger(name)
    if not this_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        this_logger.addHandler(console_handler)

        # detailed_handler = logging.StreamHandler(sys.stdout)
        # detailed_handler.setLevel(logging.DEBUG)
        # detailed_formatter = logging.Formatter(DETAILED_LOG_FORMAT)
        # detailed_handler.setFormatter(detailed_formatter)
        # this_logger.addHandler(detailed_handler)

    this_logger.setLevel(logging.DEBUG)
    this_logger.propagate = True
    return this_logger
