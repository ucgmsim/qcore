"""
A class to retrieve versioned files.
Defaults are available from the config files
"""

from config import qconfig
from os import path

default_bin_loc = qconfig['tools_dir']
default_hf_ver = "5.4.5"
default_lf_ver = "3.0.4"
default_genslip_ver = "3.3"

hf_binmod_name_str = "hb_high_v{}_binmod"
hf_np2mm_name_str = "hb_high_v{}_np2mm+"
lf_emod3d_name_str = "emod3d-mpi_v{}"
genslip_name_str = "genslip-{}"


def get_hf_binmod(version=default_hf_ver, bin_loc=default_bin_loc):
    return path.join(bin_loc, hf_binmod_name_str.format(version))


def get_hf_np2mm(version=default_hf_ver, bin_loc=default_bin_loc):
    return path.join(bin_loc, hf_np2mm_name_str.format(version))


def get_lf_bin(version=default_lf_ver, bin_loc=default_bin_loc):
    return path.join(bin_loc, lf_emod3d_name_str.format(version))


def get_genslip_bin(version=default_genslip_ver, bin_loc=default_bin_loc):
    return path.join(bin_loc,  genslip_name_str.format(version))


def get_unversioned_bin(bin_name, bin_loc=default_bin_loc):
    return path.join(bin_name, bin_loc)

