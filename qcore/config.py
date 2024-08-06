"""
Module for determining and loading machine-specific configurations.

This module provides functionality for identifying the machine type based
on its hostname and retrieving the corresponding configuration settings. The
module also defines key configuration parameters through an enumeration for
better clarity and type safety.

Functions
---------
determine_machine_config
  Determine the machine type and construct the path to the machine's
  configuration JSON file based on the hostname.
get_machine_config
  A function that loads the machine configuration from a JSON file. It can
  use a default path based on the machine type or an optional overridden path.

Usage
-----
1. Use `determine_machine_config` to identify the machine and get the path to its configuration file.
2. Use `get_machine_config` to load the configuration data from the JSON file.
3. The `ConfigKeys` enumeration can be used to access configuration keys in a type-safe manner.
"""

import re
from enum import Enum, auto
from json import load
from pathlib import Path
from platform import node
from typing import Optional, Tuple, TypedDict


class ConfigDict(TypedDict):
    """A type-annotation describing the machine config keys and their types."""

    tools_dir: str
    cores_per_node: int
    memory_per_core: float
    MAX_JOB_WCT: int
    MAX_JOB_WCT: int
    MAX_CH_PER_JOB: int


MACHINE_MAPPINGS = {
    r"ni\d{4}|maui.*": "maui",
    r"wb\d{4}|mahuika.*": "mahuika",
    r".*stampede.*": "stampede2",
    r"(login|node|nurion).*": "nurion",
}


def determine_machine_config(hostname: str = node()) -> Tuple[str, str]:
    """Determines the machine name eg: nodes ni0002 and maui01 belong to maui.

    Parameters
    ----------
    hostname : str
        The hostname of the machine.

    Returns
    -------
    Tuple[str, str]
        A tuple of the machine this host belongs to and path to the machine config JSON file.
    """
    for hostname_pattern, machine in MACHINE_MAPPINGS.items():
        if re.match(hostname_pattern, hostname):
            break
    else:
        machine = "local"

    basename = f"machine_{machine}.json"

    config_path = Path(__file__).resolve().parent / "configs" / basename
    return machine, str(config_path)


def get_machine_config(
    hostname: str = node(), config_path: Optional[Path | str] = None
) -> ConfigDict:
    """Get the current machine config.

    Parameters
    ----------
    hostname : str, defaults to $HOSTNAME
        The hostname to get machine config for.
    config_path : Optional[Path | str]
        An optional path to override the machine config.

    Returns
    -------
    ConfigDict
        The dictionary containing the machine config.
    """
    if config_path is None:
        _, config_path = determine_machine_config(hostname)
    with open(config_path, "r") as machine_config_file:
        return load(machine_config_file)


class ConfigKeys(Enum):
    """The configuration keys supported in the config. See also ConfigDict."""

    tools_dir = auto()
    cores_per_node = auto()
    memory_per_core = auto()
    MAX_JOB_WCT = auto()
    MAX_NODES_PER_JOB = auto()
    MAX_CH_PER_JOB = auto()


host, host_config_path = determine_machine_config()
qconfig: ConfigDict = get_machine_config(config_path=host_config_path)
module_requirments = str(
    Path(qconfig[ConfigKeys.tools_dir.name]) / "module_requirements"
)
