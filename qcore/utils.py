"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""

import os
from collections import OrderedDict
from collections.abc import Mapping
from shutil import rmtree

import yaml


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


def compare_versions(version1: str, version2: str, split_char: str = ".") -> int:
    """Compare two version strings.

    Comparison is made on the individual parts of each version. Where the
    number of parts differs, i.e. comparing 1.1 and 1, the smaller version
    is padded with zeros before comparison.

    Parameters
    ----------
    version1 : str
        The first version string to check.
    version2 : str
        The second version string to check.
    split_char : str
        The version separator.

    Returns
    -------
    int
        Returns 1 if version1 is newer than version 2, -1 if version2 is
        newer than version1 and 0 otherwise.

    Examples
    --------
    >>> compare_versions('1.0.0', '1')
    0
    >>> compare_versions('1.0.1', '1')
    1
    >>> compare_versions('1.0.1', '1.1')
    -1
    """
    invalid_version_characters = f"[^0-9{re.escape(split_char)}]"
    parts1 = [
        int(part)
        for part in re.sub(invalid_version_characters, "", version1).split(split_char)
    ]
    parts2 = [
        int(part)
        for part in re.sub(invalid_version_characters, "", version2).split(split_char)
    ]
    max_length = max(len(parts1), len(parts2))
    parts1.extend((max_length - len(parts1)) * [0])
    parts2.extend((max_length - len(parts2)) * [0])

    if parts1 > parts2:
        return 1
    if parts1 < parts2:
        return -1
    return 0
