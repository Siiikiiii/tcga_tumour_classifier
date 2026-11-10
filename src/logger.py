ger · PY
# logger.py
"""
Applies the logging configuration defined in settings.LOGGING_CONFIG.

Configuration is applied by calling setup_logging() at the start of each
program (train.py, app.py) rather than when this module is imported, so
importing project modules does not change logging set-up as a side effect;
it happens only when a program actually runs.
"""

import logging.config

from src.settings import LOG_DIR, LOGGING_CONFIG


# --------------------------------------------------
# LOGGING ACTIVATION FUNCTION
# --------------------------------------------------
def setup_logging():
    """
    Apply the logging configuration.

    Call once at program start-up. Python's logging is process-wide, so
    after this call every module can simply use logging.getLogger(__name__):
    loggers named "src.*" pick up the "src" settings and handlers, and all
    others (third-party libraries) fall back to the root logger.
    """

    # CHANGED: create the log folder here (moved from settings.py), because
    # the file handler fails if it doesn't exist, e.g. on a fresh clone from
    # GitHub, which doesn't store empty folders.
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(LOGGING_CONFIG)