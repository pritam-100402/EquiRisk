"""
src/utils/logging_config.py

One place to configure logging for the whole project.

Every module already does `logger = logging.getLogger("equirisk.<area>")`,
which is the right pattern -- named loggers form a hierarchy under the
"equirisk" root, so configuring that one root configures all of them.
What the modules should NOT each do is call logging.basicConfig() at
import time: whichever module imports first wins, the rest are silent
no-ops, and the resulting format depends on import order.

Entrypoints (the orchestrator, the CLI script, notebooks) call
setup_logging() once. Library modules just get their logger and log.

Third-party libraries are turned down separately -- py4j in particular
emits a wall of DEBUG traffic per Spark call that buries everything else.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_NOISY_LIBRARIES = {
    "py4j": logging.WARNING,
    "py4j.java_gateway": logging.WARNING,
    "botocore": logging.WARNING,
    "boto3": logging.WARNING,
    "s3transfer": logging.WARNING,
    "urllib3": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "matplotlib": logging.WARNING,
}


def setup_logging(level: int = logging.INFO, quiet_libraries: bool = True) -> logging.Logger:
    """Configure the 'equirisk' logger tree. Safe to call more than once
    -- existing handlers are cleared first, so a second call replaces the
    configuration rather than duplicating every log line (the usual
    symptom of re-running a notebook cell).

    Returns the 'equirisk' logger, mostly so entrypoints can log directly
    without a second import.
    """
    root = logging.getLogger("equirisk")

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(handler)
    root.setLevel(level)

    # Don't hand records to the global root logger as well -- otherwise a
    # stray logging.basicConfig() anywhere prints everything twice.
    root.propagate = False

    if quiet_libraries:
        for name, lib_level in _NOISY_LIBRARIES.items():
            logging.getLogger(name).setLevel(lib_level)

    return root


def get_logger(name: str) -> logging.Logger:
    """Convenience for modules: get_logger("etl.sentiment") returns the
    logger named "equirisk.etl.sentiment". Purely optional -- calling
    logging.getLogger("equirisk.etl.sentiment") directly is equivalent."""
    return logging.getLogger(f"equirisk.{name}")
