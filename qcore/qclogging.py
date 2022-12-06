import logging
from logging.handlers import MemoryHandler
import sys
from typing import Union

from qcore.constants import ProcessType

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

GENERAL_THREADED_LOGGING_MESSAGE_FORMAT = "%(levelname)8s -- %(asctime)s - %(threadName)s - %(module)s.%(funcName)s - %(message)s"
general_threaded_formatter = logging.Formatter(GENERAL_LOGGING_MESSAGE_FORMAT)

TASK_LOGGING_MESSAGE_FORMAT = (
    "%(levelname)8s -- %(asctime)s - %(module)s.%(funcName)s - {}.{} - %(message)s"
)
TASK_THREADED_LOGGING_MESSAGE_FORMAT = "%(levelname)8s -- %(asctime)s - %(threadName)s - %(module)s.%(funcName)s - {}.{} - %(message)s"

REALISATION_LOGGING_MESSAGE_FORMAT = (
    "%(levelname)8s -- %(asctime)s - %(module)s.%(funcName)s - {} - %(message)s"
)
REALISATION_THREADED_LOGGING_MESSAGE_FORMAT = "%(levelname)8s -- %(asctime)s - %(threadName)s - %(module)s.%(funcName)s - {} - %(message)s"


def get_basic_logger():
    basic_logger = logging.getLogger("Basic")
    basic_logger.setLevel(logging.INFO)
    return basic_logger


def get_logger(
    name: Union[str, None] = DEFAULT_LOGGER_NAME, threaded=False, stdout_printer=True
) -> logging.Logger:
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
    if len(logger.handlers) > 0:
        return logger
    logger.setLevel(logging.DEBUG)

    if stdout_printer:
        logger.addHandler(create_stdout_handler(name))

    return logger


def get_realisation_logger(
    old_logger: logging.Logger, realisation: str
) -> logging.Logger:
    """Creates a new logger that logs the realisation.
    The logger passed in is effectively duplicated and log messages are saved to the same file as the original logger.
    :param old_logger: Logger the new instance is to be based on
    :param realisation: The name of the realisation this logger is for
    :param threaded: If the logger is operating in a thread then record the name of the thread
    :return: The new logger object
    """
    new_logger = logging.getLogger(realisation)
    if len(new_logger.handlers) > 0:
        return new_logger
    new_logger.setLevel(logging.DEBUG)

    if old_logger.name.startswith(THREADED):
        task_formatter = logging.Formatter(
            REALISATION_THREADED_LOGGING_MESSAGE_FORMAT.format(realisation)
        )
    else:
        task_formatter = logging.Formatter(
            REALISATION_LOGGING_MESSAGE_FORMAT.format(realisation)
        )

    for handler in duplicate_handlers(old_logger.handlers, task_formatter):
        new_logger.addHandler(handler)

    new_logger.addHandler(create_stdout_handler(old_logger.name))

    return new_logger


def get_task_logger(
    old_logger: logging.Logger, realisation: str, process_type: int
) -> logging.Logger:
    """Creates a new logger that logs the realisation and process type.
    The logger passed in is effectively duplicated and log messages are saved to the same file as the original logger.
    :param old_logger: Logger the new instance is to be based on
    :param realisation: The name of the realisation this logger is for
    :param process_type: The type of process these logs are generated from
    :return: The new logger object
    """

    process_name = ProcessType(process_type).str_value

    new_logger = logging.getLogger("{}.{}".format(realisation, process_name))
    if len(new_logger.handlers) > 0:
        return new_logger
    new_logger.setLevel(logging.DEBUG)

    if old_logger.name.startswith(THREADED):
        task_formatter = logging.Formatter(
            TASK_THREADED_LOGGING_MESSAGE_FORMAT.format(realisation, process_name)
        )
    else:
        task_formatter = logging.Formatter(
            TASK_LOGGING_MESSAGE_FORMAT.format(realisation, process_name)
        )

    for handler in duplicate_handlers(old_logger.handlers, task_formatter):
        new_logger.addHandler(handler)

    new_logger.addHandler(create_stdout_handler(old_logger.name))

    return new_logger


def add_general_file_handler(logger: logging.Logger, file_path: str):
    """
    Adds a file handler to the logger using the given file_path
    If there are any Memory handler attached to the logger they are automatically drained in to the new file handler
    :param logger: The logger object
    :param file_path: The path to the file to be used. Will be appended to if it already exists
    """
    file_out_handler = logging.FileHandler(file_path)
    if logger.name.startswith(THREADED):
        file_out_handler.setFormatter(general_threaded_formatter)
    else:
        file_out_handler.setFormatter(general_formatter)

    logger.addHandler(file_out_handler)


def add_buffer_handler(
    logger: logging.Logger,
    buffer_size: int = 100,
    flush_level: int = 1000,
    file_name: str = None,
):
    """
    Adds a buffer handler to the logger.
    Useful for log files that are written to the output directory if that directory does not exist yet
    :param logger: The logger object
    :param buffer_size: The number of messages to buffer before a flush is forced
    :param flush_level: The minimum level of a message to cause the handler to be flushed early.
    Defaults to a value that won't be reached by normal log levels to prevent premature flushing
    :param file_name: The name of the log file to be used when it is available
    """
    # Flush level should be high enough that flush is not called for regular messages
    buffer_handler = MemoryHandler(buffer_size, flushLevel=flush_level)
    if logger.name.startswith(THREADED):
        buffer_handler.setFormatter(general_threaded_formatter)
    else:
        buffer_handler.setFormatter(general_formatter)

    if file_name is not None:
        file_out_handler = logging.FileHandler(file_name, delay=True)
        if logger.name.startswith(THREADED):
            file_out_handler.setFormatter(general_threaded_formatter)
        else:
            file_out_handler.setFormatter(general_formatter)

        buffer_handler.setTarget(file_out_handler)

    logger.addHandler(buffer_handler)


def remove_buffer_handler(logger: logging.Logger):
    for handler in logger.handlers[:]:
        if isinstance(handler, MemoryHandler):
            popped_handler = handler.target
            if popped_handler is not None:
                logger.addHandler(popped_handler)
            handler.close()
            logger.handlers.remove(handler)


def create_stdout_handler(logger_name):
    task_print_handler = logging.StreamHandler(sys.stdout)
    task_print_handler.setLevel(logging.INFO)
    if logger_name.startswith(THREADED):
        task_print_handler.setFormatter(stdout_threaded_formatter)
    else:
        task_print_handler.setFormatter(stdout_formatter)
    # If the message level ends in 1 do not print it to stdout
    task_print_handler.addFilter(lambda record: (record.levelno % 10) != 1)
    return task_print_handler


def duplicate_handlers(old_handlers, formatter):
    log_files = []
    new_handlers = []
    for handler in old_handlers:
        if isinstance(handler, logging.FileHandler):
            log_name = handler.baseFilename
            if log_name in log_files:
                continue
            log_files.append(log_name)
            task_file_out_handler = logging.FileHandler(log_name)
            task_file_out_handler.setFormatter(formatter)
            task_file_out_handler.setLevel(handler.level)
            new_handlers.append(task_file_out_handler)

        if isinstance(handler, MemoryHandler) and isinstance(
            handler.target, logging.FileHandler
        ):
            log_name = handler.target.baseFilename
            if log_name in log_files:
                continue
            log_files.append(log_name)

            task_file_out_handler = logging.FileHandler(log_name, delay=True)
            task_file_out_handler.setFormatter(formatter)
            task_file_out_handler.setLevel(handler.level)

            task_mem_handler = MemoryHandler(
                handler.capacity, flushLevel=handler.flushLevel
            )
            task_mem_handler.setFormatter(formatter)
            task_mem_handler.setLevel(handler.level)
            task_mem_handler.setTarget(task_file_out_handler)
            new_handlers.append(task_mem_handler)

    return new_handlers


def clean_up_logger(logger: logging.Logger):
    for handler in logger.handlers[::-1]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        if isinstance(handler, MemoryHandler):
            handler.close()
        logger.removeHandler(handler)


def set_stdout_level(logger: logging.Logger, level: int):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)
