
import json
import os
import platform

hostname = platform.node()
if hostname.startswith("ni") and len(hostname) == 8:
    # maui
    basename = os.path.join('machine_config', 'config_maui.json')
elif hostname.startswith("mahuika") and len(hostname) == 6:
    # mahuika
    basename = os.path.join('machine_config', 'config_mahuika.json')
else:
    # default
    basename = 'config.json'

config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        basename)

with open(config_file, 'r') as f:
    qconfig = json.load(f)


def get_host_config():
    """
    Determine the actual hostname eg.ni0002--> maui; maui01-->maui
    :return: actual hostname, config josn file path
    """
    host_name = platform.node()
    if (host_name.startswith("ni") and len(host_name) == 8) or host_name.startswith('maui'):  # maui
        actual_host_name = "maui"
        base_name = os.path.join('machine_config', 'config_maui.json')

    elif host_name.startswith("mahuika") and len(host_name) == 6:  # mahuika
        actual_host_name = "mahuika"
        base_name = os.path.join('machine_config', 'config_mahuika.json')

    else:  # default
        actual_host_name = "default"
        base_name = "config.json"

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_name)
    return actual_host_name, config_path


def get_tools_dir(bin_name, version='3.0.4-gcc'):
    """
    first get opt_dir from machine config based on host_name
    then dynamically determine the tools_dir based on host_name, bin_name and version
    :param bin_name: emod3d, hf, bb
    :param version: eg. 3.0.3-gcc
    :return: tools_dir path
    """
    host_name, config_path = get_host_config()
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if host_name == 'maui' or host_name == 'mahuika':
        opt_dir = config_data['opt_dir']
        tools_dir = os.path.join(opt_dir, host_name, bin_name, version, 'bin')
    else:  # default
        tools_dir = config_data['tools_dir']

    return tools_dir
