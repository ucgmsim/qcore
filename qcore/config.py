from enum import Enum, auto
from json import load
from os.path import join, abspath, dirname
from platform import node


class __KnownMachines(Enum):
    # Enum intended for local use only
    # The platform config in slurm gm workflow creates a dynamic machines enum
    local = auto()
    maui = auto()
    mahuika = auto()
    stampede2 = auto()
    nurion = auto()


def determine_machine_config(hostname=node()):
    """
    Manages multiple configurations for different machines.
    Determines the machine name eg: nodes ni0002 and maui01 belong to maui.
    :return: machine name, config file
    """

    if (hostname.startswith("ni") and len(hostname) == 8) or hostname.startswith(
        __KnownMachines.maui.name
    ):
        machine = __KnownMachines.maui.name
    elif (hostname.startswith("wb") and len(hostname) == 6) or hostname.startswith(
        __KnownMachines.mahuika.name
    ):
        machine = __KnownMachines.mahuika.name
    elif hostname.find("stampede") > -1:
        machine = __KnownMachines.stampede2.name
    elif (
        hostname.startswith("login")
        or hostname.startswith("node")
        or hostname.startswith(__KnownMachines.nurion.name)
    ):
        machine = __KnownMachines.nurion.name
    else:
        machine = __KnownMachines.local.name

    basename = f"machine_{machine}.json"

    config_path = join(dirname(abspath(__file__)), "configs", basename)
    return machine, config_path


def get_machine_config(hostname=node(), config_path=None):
    if config_path is None:
        _, config_path = determine_machine_config(hostname)
    with open(config_path, "r") as machine_config_file:
        return load(machine_config_file)


class ConfigKeys(Enum):
    GMT_DATA = auto()
    tools_dir = auto()
    cores_per_node = auto()
    MAX_JOB_WCT = auto()
    MAX_NODES_PER_JOB = auto()


host, host_config_path = determine_machine_config()
qconfig = get_machine_config(config_path=host_config_path)
