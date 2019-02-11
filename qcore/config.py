import json
import os
import platform


def determine_machine_config():
    """
    Manages multiple configurations for different machines.
    Determines the machine name eg: nodes ni0002 and maui01 belong to maui.
    :return: machine name, config file
    """
    hostname = platform.node()
    if (hostname.startswith("ni") and len(host_name) == 8) or host_name.startswith('maui'):
        machine = "maui"
        basename = os.path.join('machine_config', 'config_maui.json')

    elif (host_name.startswith("wb") and len(host_name) == 6) or host_name.startswith("mahuika"):
        machine = "mahuika"
        basename = os.path.join('machine_config', 'config_mahuika.json')

    else:
        machine = "default"
        basename = "config.json"

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), basename)
    return machine, config_path


host, config_file = determine_machine_config()

with open(config_file, 'r') as f:
    qconfig = json.load(f)

