from json import load
from os.path import join, abspath, dirname
from platform import node


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
    elif (hostname.startswith("wb") and len(hostname) == 6) or hostname.startswith(
        "mahuika"
    ):
        machine = "mahuika"
    elif hostname.find("stampede") > -1:
        machine = "stampede2"
    elif (
        hostname.startswith("login")
        or hostname.startswith("node")
        or hostname.startswith("nurion")
    ):
        machine = "nurion"
    else:
        machine = "local"

    basename = f"machine_{machine}.json"

    config_path = join(dirname(abspath(__file__)), "configs", basename)
    return machine, config_path


def get_machine_config(hostname=node(), config_path=None):
    if config_path is None:
        _, config_path = determine_machine_config(hostname)
    with open(config_path, "r") as machine_config_file:
        return load(machine_config_file)


host, host_config_path = determine_machine_config()
qconfig = get_machine_config(config_path=host_config_path)
