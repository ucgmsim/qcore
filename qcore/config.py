from enum import Enum
from json import load
from os.path import join, abspath, dirname
from platform import node

from qcore.constants import PLATFORM_CONFIG


def determine_machine_config(hostname=node()):
    """
    Manages multiple configurations for different machines.
    Determines the machine name eg: nodes ni0002 and maui01 belong to maui.
    :return: machine name, config file
    """

    if (hostname.startswith("ni") and len(hostname) == 8) or hostname.startswith(
        "maui"
    ):
        machine = "maui"
        basename = "config_maui.json"

    elif (hostname.startswith("wb") and len(hostname) == 6) or hostname.startswith(
        "mahuika"
    ):
        machine = "mahuika"
        basename = "config_mahuika.json"

    elif hostname.find("stampede") > -1:
        machine = "stampede2"
        basename = "config_stampede2.json"

    elif (
        hostname.startswith("login")
        or hostname.startswith("node")
        or hostname.startswith("nurion")
    ):
        machine = "nurion"
        basename = "config_nurion.json"

    else:
        machine = "default"
        basename = "config_default.json"

    config_path = join(dirname(abspath(__file__)), "machine_config", basename)
    return machine, config_path


def determine_platform_config(hostname=determine_machine_config()[0]):
    if hostname == "maui" or hostname == "mahuika":
        hpc_platform = "nesi"
        basename = "config_nesi.json"

    elif hostname == "stampede2":
        hpc_platform = "tacc"
        basename = "config_tacc.json"

    elif hostname == "nurion":
        hpc_platform = "kisti"
        basename = "config_kisti.json"

    elif hostname == "default":
        hpc_platform = "local"
        basename = "config_bash.json"

    else:
        raise ValueError("Unexpected host given")

    config_path = join(dirname(abspath(__file__)), "machine_config", basename)
    return hpc_platform, config_path


def get_machine_config(hostname=node(), config_path=None):
    if config_path is None:
        _, config_path = determine_machine_config(hostname)
    with open(config_path, "r") as machine_config_file:
        return load(machine_config_file)


host, host_config_path = determine_machine_config()
qconfig = get_machine_config(config_path=host_config_path)

platform, platform_config_path = determine_platform_config(host)
platform_config = get_machine_config(config_path=platform_config_path)
errors = set(platform_config.keys()).symmetric_difference(
    set([key.name for key in PLATFORM_CONFIG])
)
if errors:
    missing_keys = []
    extra_keys = []
    for key in errors:
        if key in platform_config:
            extra_keys.append(key)
        else:
            missing_keys.append(key)
    message = (
        f"There were some errors with the platform config file {platform_config_path}."
    )
    if missing_keys:
        message += f" Missing keys: {', '.join(missing_keys)}."
    if extra_keys:
        message += f" Additional keys found: {', '.join(extra_keys)}."
    raise ValueError(message)

# Dynamically generate the HPC enum
HPC = Enum("HPC", platform_config[PLATFORM_CONFIG.AVAILABLE_MACHINES.name], module=__name__)
