"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""

from shutil import rmtree
import os
import imp

def setup_dir(directory, empty = False):
    """
    Make sure a directory exists, optionally make sure it is empty.
    directory: path to directory
    empty: make sure directory is empty

    :param directory:
    :param empty:
    :return:
    """
    if os.path.exists(directory) and empty:
            rmtree(directory)
    if not os.path.exists(directory):
        # multi processing safety (not useful with empty set)
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise

def load_py_cfg(f_path):
	"""
	loads a python configuration file to a dictionary

        if you want to preserve the import params functionality, locals().update(cfg_dict) converts the returned dict to local variables.

	:param f_path: path to configuration file
	:return: dict of parameters
	"""
	with open(f_path) as f:
		module = imp.load_module('params', f, f_path, ('.py', 'r', imp.PY_SOURCE))
		cfg_dict = module.__dict__
		
	return cfg_dict
