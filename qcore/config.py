
import json
import os
import platform

hostname = platform.node()
if hostname.startswith("ni") and len(hostname) == 8:
    host = "maui"
    basename = os.path.join('machine_config', 'config_maui.json')
elif hostname.startswith("wb") and len(hostname) == 6:
    host = "mahuika"
    basename = os.path.join('machine_config', 'config_mahuika.json')
else:
    # default
    host = "unknown"
    basename = 'config.json'

config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        basename)

with open(config_file, 'r') as f:
    qconfig = json.load(f)
