"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Any, Union

import yaml


def load_yaml(yaml_file: Union[Path, str]) -> Any:
    """Load YAML from a file.

    *DO NOT USE*. This function exists for backwards compatibility only. Just
    use yaml.safe_load instead.

    Parameters
    ----------
    yaml_file : Union[Path, str]
        The filepath of the YAML file.

    Returns
    -------
    Any
        The contents of the YAML file as a Python object (usually, a dictionary).
    """
    with open(yaml_file, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def dump_yaml(object: Any, output_name: Union[Path, str]):
    """Dump an object to a YAML file.

    *DO NOT USE*. This function exists for backwards compatibility only. Just
    use yaml.safe_dump instead.

    Parameters
    ----------
    object : Any
        The object to dump.
    output_name : Union[Path, str]
        The filepath to dump to.
    """
    with open(output_name, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(object, yaml_file)


def setup_dir(directory: str, empty: bool = False):
    """Ensure a directory exists and, optionally, that it is empty.

    Parameters
    ----------
    directory : str
        The directory to check.
    empty : bool
        If True, check if the directory is empty.
    """
    if os.path.exists(directory) and empty:
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        # multi processing safety (not useful with empty set)
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise


def compare_versions(version1: str, version2: str, split_char: str = ".") -> int:
    """Compare two version strings.

    Comparison is made on the individual parts of each version. Where the
    versions are equivalent but the number of parts differs, i.e. comparing
    1.0 and 1, the longer version string is considered newer.

    Parameters
    ----------
    version1 : str
        The first version string to check.
    version2 : str
        The second version string to check.
    split_char : str
        The version separator.

    Returns
    -------
    int
        Returns 1 if version1 is newer than version 2, -1 if version2 is
        newer than version1 and 0 otherwise.

    Examples
    --------
    >>> compare_versions('1.0.0', '1')
    0
    >>> compare_versions('1.0.1', '1')
    1
    >>> compare_versions('1.0.1', '1.1')
    -1
    """
    invalid_version_characters = f"[^0-9{re.escape(split_char)}]"
    parts1 = [
        int(part)
        for part in re.sub(invalid_version_characters, "", version1).split(split_char)
    ]
    parts2 = [
        int(part)
        for part in re.sub(invalid_version_characters, "", version2).split(split_char)
    ]
    max_length = max(len(parts1), len(parts2))

    if parts1[:max_length] > parts2[:max_length]:
        return 1
    if parts1[:max_length] < parts2[:max_length]:
        return -1

    if len(parts1) > len(parts2):
        return 1
    if len(parts2) > len(parts1):
        return -1

    return 0
