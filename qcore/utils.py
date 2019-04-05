"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""


import os
import imp
import yaml
from shutil import rmtree
from collections import OrderedDict
from collections import Mapping


class DotDictify(dict):
    """
    Construct an dictionary object whose values can also be accessed by 'dot'
    eg. d.k; d.k1.k2
    """
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDictify):
            value = DotDictify(value)
        super(DotDictify, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotDictify.MARKER)
        if found is DotDictify.MARKER:
            found = DotDictify()
            super(DotDictify, self).__setitem__(key, found)
        return found

    def __call__(self, *args, **kwargs):
        return self

    def __getstate__(self):
        return self.__dict__

    __setattr__, __getattr__ = __setitem__, __getitem__


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
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def load_yaml(yaml_file, obj_type=dict):
    """
    load yaml file into a OrderedDict/dict
    :param yaml_file: path to yaml file
    :param obj_type: =OrderedDict to load yaml in order;
                     =dict to load in random order
    :return: OrderedDict/dict
    """
    with open(yaml_file, 'r') as stream:
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
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

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
    with open(output_name, 'w') as yaml_file:
        ordered_dump(input_dict, yaml_file, Dumper=yaml.SafeDumper, representer=obj_type, default_flow_style=False)
        # yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
        # yaml.dump(input_dict, yaml_file, default_flow_style=False)


def update(d, *u):
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
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
    return d


def load_sim_params(sim_yaml_path, load_fault=True, load_root=True, load_vm=True):
    """
    load all necessary params for a single simulation
    :param sim_yaml_path: path to sim_params.yaml
    :param load_fault: to load fault_params.yaml or not
    :param load_root: to load root_params.yaml or not
    :param load_vm: to load vm_params.yaml or not
    :return: a DotDictify object that contains all necessary params for a single simulation
    """
    sim_params = load_yaml(sim_yaml_path)
    fault_params = {}
    root_params ={}
    vm_params = {}
    if load_root or load_vm and not load_fault:
        load_fault = True   #root/vm_yamlpath in fault_yaml
    if load_fault:
        fault_params = load_yaml(sim_params['fault_yaml_path'])
    if load_root:
        root_params = load_yaml(fault_params['root_yaml_path'])
    if load_vm:
        vm_params = load_yaml(os.path.join(fault_params['vel_mod_dir'], 'vm_params.yaml'))
    return DotDictify(update(vm_params, root_params, fault_params, sim_params))


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
        module = imp.load_module('params', f, f_path, ('.py', 'r', imp.PY_SOURCE))
        cfg_dict = module.__dict__

    return cfg_dict
