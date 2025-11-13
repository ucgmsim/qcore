"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""

from pathlib import Path
from typing import Any, Union
from warnings import deprecated

import yaml


@deprecated("use yaml.safe_load")
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
