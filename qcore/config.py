import json
import os
import platform

def get_host_config():
    """
    Determine the actual hostname eg.ni0002--> maui; maui01-->maui
    :return: actual hostname, config josn file path
    """
    host_name = platform.node()
    if (host_name.startswith("ni") and len(host_name) == 8) or host_name.startswith('maui'):  # maui
        actual_host_name = "maui"
        base_name = os.path.join('machine_config', 'config_maui.json')

    elif (host_name.startswith("wb") and len(host_name) == 6) or host_name.startswith("mahuika"):  # mahuika
        actual_host_name = "mahuika"
        base_name = os.path.join('machine_config', 'config_mahuika.json')

    else:  # default
        actual_host_name = "default"
        base_name = "config.json"

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_name)
    return actual_host_name, config_path


host, config = get_host_config()
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        config)

with open(config_file, 'r') as f:
    qconfig = json.load(f)