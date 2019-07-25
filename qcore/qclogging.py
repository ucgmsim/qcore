import logging
import sys

VERYVERBOSE = logging.DEBUG // 2

logging.addLevelName(VERYVERBOSE, "VERY_VERBOSE")

NOPRINTCRITICAL = logging.CRITICAL + 1
NOPRINTERROR = logging.ERROR + 1
NOPRINTWARNING = logging.WARNING + 1
NOPRINTINFO = logging.INFO + 1
NOPRINTDEBUG = logging.DEBUG + 1

logging.addLevelName(NOPRINTCRITICAL, "NO_PRINT_CRITICAL")
logging.addLevelName(NOPRINTERROR, "NO_PRINT_ERROR")
logging.addLevelName(NOPRINTWARNING, "NO_PRINT_WARNING")
logging.addLevelName(NOPRINTINFO, "NO_PRINT_INFO")
logging.addLevelName(NOPRINTDEBUG, "NO_PRINT_DEBUG")

THREADED = "THREADED"
DEFAULT_LOGGER_NAME = "default_logger"

STDOUT_MESSAGE_FORMAT = "%(asctime)s - %(message)s"
stdout_formatter = logging.Formatter(STDOUT_MESSAGE_FORMAT)

STDOUT_THREADED_MESSAGE_FORMAT = "%(asctime)s - %(threadName)s - %(message)s"
stdout_threaded_formatter = logging.Formatter(STDOUT_THREADED_MESSAGE_FORMAT)

GENERAL_LOGGING_MESSAGE_FORMAT = (
    "%(levelname)8s -- %(asctime)s - %(module)s.%(funcName)s - %(message)s"
)
general_formatter = logging.Formatter(GENERAL_LOGGING_MESSAGE_FORMAT)

GENERAL_THREADED_LOGGING_MESSAGE_FORMAT = (
    "%(levelname)8s -- %(asctime)s - %(threadName)s - %(module)s.%(funcName)s - %(message)s"
)
general_threaded_formatter = logging.Formatter(GENERAL_LOGGING_MESSAGE_FORMAT)


def get_logger(name: str = DEFAULT_LOGGER_NAME, threaded=False, stdout_printer=True) -> logging.Logger:
    """
    Creates a logger and an associated handler to print messages over level INFO to stdout.
    The handler is configured such that messages will not be printed if their underlying level value ends in 1, this is
    mostly used for logging fatal exceptions that will be printed to stdout/stderr anyway
    :param name: Name of the logger. If a logger with that name already exists it will be returned
    :param threaded: If the logger is operating in a thread then record the name of the thread
    :return: The logger object
    """
    if name is not None and threaded:
        name = "{}_{}".format(THREADED, name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if stdout_printer:
        print_handler = logging.StreamHandler(sys.stdout)
        print_handler.setLevel(logging.INFO)
        if threaded:
            print_handler.setFormatter(stdout_threaded_formatter)
        else:
            print_handler.setFormatter(stdout_formatter)

        # If the message level ends in 1 do not print it to stdout
        print_handler.addFilter(lambda record: (record.levelno % 10) != 1)

        logger.addHandler(print_handler)

    return logger


def add_general_file_handler(
    logger: logging.Logger, file_path: str
):
    """
    Adds a file handler to the logger using the given file_path
    :param logger: The logger object
    :param file_path: The path to the file to be used. Will be appended to if it already exists
    """
    file_out_handler = logging.FileHandler(file_path)
    if logger.name.startswith(THREADED):
        file_out_handler.setFormatter(general_threaded_formatter)
    else:
        file_out_handler.setFormatter(general_formatter)

    logger.addHandler(file_out_handler)


def get_basic_logger():
    basic_logger = logging.getLogger("Basic")
    basic_logger.setLevel(logging.INFO)
    return basic_logger


def clean_up_logger(logger: logging.Logger):
    for handler in logger.handlers[::-1]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.handlers.remove(handler)


def set_stdout_level(logger: logging.Logger, level: int):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)
