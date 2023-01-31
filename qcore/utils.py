"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""


import os
import imp
import yaml
from shutil import rmtree
from collections import OrderedDict
from collections.abc import Mapping


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    :param stream: yaml file path
    :param Loader: yaml loader
    :param object_pairs_hook: =OrderedDict to load file in order;
                              =dict to load in random order
    :return: OrderedDict
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def load_yaml(yaml_file, obj_type=dict):
    """
    load yaml file into a OrderedDict/dict
    :param yaml_file: path to yaml file
    :param obj_type: =OrderedDict to load yaml in order;
                     =dict to load in random order
    :return: OrderedDict/dict
    """
    with open(yaml_file, "r") as stream:
        return ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=obj_type)


def ordered_dump(data, stream, Dumper=yaml.Dumper, representer=OrderedDict, **kwds):
    """
    write data dict into a yaml file.
    :param: data: input dict
    :param stream: output yaml file
    :param Dumper: yaml.Dumper
    :param representer: =OrderedDict to write in order;
                        =dict to write in random order
    :param kwds: optional args for writing a yaml file;
                 eg.default_flow_style=False
    :return: yaml file
    """

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
        )

    OrderedDumper.add_representer(representer, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def dump_yaml(input_dict, output_name, obj_type=dict):
    """
    :param input_dict: input dict to write into a yaml file
    :param output_name: output path (name inclusive) of the yaml file
    :param obj_type: =OrderedDict to write in order;
                     =dict to write in random order
    :return:
    """
    with open(output_name, "w") as yaml_file:
        ordered_dump(
            input_dict,
            yaml_file,
            Dumper=yaml.SafeDumper,
            representer=obj_type,
            default_flow_style=False,
        )
        # yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        # yaml.dump(input_dict, yaml_file, default_flow_style=False)


def _update_params(d, *u):
    """
    prevents removal of keys in a nested dict
    Note the same key will still be overwritten
    Last dict in *u would preserve all its keys
    eg.a = {hf: {hf_dt: 1, x: 2}}
       b = {hf: {hf_dt: 3, y: 4}}
    with builtin dict.update, a.update(b) == b = {hf: {hf_dt: 3, y: 4}}
    with this custom update, a.update(b) = {hf: {hf_dt: 3, x: 2, y: 4}}
    :param d: original dict
    :param u: dict(s) containing updating items
    :return: updated dict d
    """
    for uu in u:
        if uu:  # if uu is not empty
            for k, v in uu.items():
                if isinstance(v, Mapping):
                    d[k] = _update_params(d.get(k, {}), v)
                else:
                    d[k] = v
    return d


def load_sim_params(sim_yaml_path=False, load_fault=True, load_root=True, load_vm=True):
    """
    load all necessary params for a single simulation
    :param sim_yaml_path: path to sim_params.yaml or a falsy value to not load it
    :param load_fault: Either True, the path to fault_params.yaml or a falsy value to not load it
    :param load_root: Either True, the path to root_params.yaml or a false value to not load it
    :param load_vm: Either True, the path to vm_params.yaml or a false value to not load it
    :return: a DotDictify object that contains all necessary params for a single simulation
    """
    sim_params = {}
    fault_params = {}
    root_params = {}
    vm_params = {}

    if load_root is True or load_vm is True and not load_fault:
        load_fault = True  # root/vm_yamlpath in fault_yaml

    if sim_yaml_path:
        sim_params = load_yaml(sim_yaml_path)
    elif load_fault is True:
        raise ValueError("For automated fault_params loading, sim_params must be set")

    if load_fault is True:
        fault_params = load_yaml(sim_params["fault_yaml_path"])
    elif load_fault:
        fault_params = load_yaml(load_fault)

    if load_root is True:
        root_params = load_yaml(fault_params["root_yaml_path"])
    elif load_root:
        root_params = load_yaml(load_root)

    if load_vm is True:
        vm_params = load_yaml(
            os.path.join(fault_params["vel_mod_dir"], "vm_params.yaml")
        )
    elif load_vm:
        vm_params = load_yaml(load_vm)

    return _update_params(vm_params, root_params, fault_params, sim_params)


def setup_dir(directory, empty=False):
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
        module = imp.load_module("params", f, f_path, (".py", "r", imp.PY_SOURCE))
        cfg_dict = module.__dict__

    return cfg_dict


def compare_versions(version1, version2, split_char="."):
    """
    Compares two version strings.
    Each string is to be split into segments by the given string.
    Each segment is only compared by numerical character value (e.g. a, b, rc are ignored)
    Ordinality is determined by the first non equal numeric segment.
    If the first argument is greater than the second then 1 is returned, if the second argunet is greater then -1 is returned.
    If an ordering has not been found by this point then the version with more segments is considered greater.
    If both have the same segments then the value 0 is returned.
    """
    parts1 = version1.split(split_char)
    parts2 = version2.split(split_char)

    num_parts = min(len(parts1), len(parts2))
    for i in range(num_parts):
        num1 = int("".join(c for c in parts1[i] if c.isnumeric()))
        num2 = int("".join(c for c in parts2[i] if c.isnumeric()))
        if num1 > num2:
            return 1
        if num2 > num1:
            return -1

    # We haven't found a match
    if len(parts1) > len(parts2):
        return 1
    if len(parts2) > len(parts1):
        return -1
    return 0


def change_file_ext(file_ffp: str, new_ext: str, excl_dot: bool = False):
    """Returns the full file path of the given file with the
    extension changed to new_ext

    If excl_dot is set, then a . is not added automatically
    """
    return os.path.join(
        os.path.dirname(file_ffp),
        os.path.splitext(os.path.basename(file_ffp))[0]
        + (f".{new_ext}" if not excl_dot else new_ext),
    )
