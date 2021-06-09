"""
Logging utils
"""
import logging

logger = logging.getLogger(__name__)


def log_and_raise(exception, msg):
    """
    Log an error and raise `exception` with the same `msg` message.

    Arguments
    ---------
    exception: exception type (ValueError, TypeError...)

    msg: str
    """
    logger.error(msg=msg)
    raise exception(msg)
