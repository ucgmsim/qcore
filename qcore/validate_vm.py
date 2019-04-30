#!/usr/bin/env python
"""
Checks EMOD3D VM folder for corectness. Compatible with Python 2.6+ and 3.0+.
Run script with VM folder location as first parameter. Returns 0 if successful.
or:
Import and use validate_vm directly.
Example of running in a bash script:
======================================
validate_vm.py /path/to/VMs/AlpineRegion
if [ $? -eq 0 ]; then
    echo success
else
    echo fail
fi
"""

import os
import sys
import argparse
from qcore.utils import load_yaml
from qcore.constants import VM_PARAMS_FILE_NAME

DEM_PATH = "/nesi/project/nesi00213/opt/Velocity-Model/Data/DEM/NZ_DEM_HD.in"

try:
    import numpy as np

    numpy = True
except ImportError:
    numpy = False


def validate_vm(vm_dir, dem_path=DEM_PATH):
    """
    Go through rules of VM directories. Return False if invalid.
    vm_dir: folder path containing VM files
    verbose: print progress
    errors: print errors and warnings to stderr
    """
    SIZE_FLOAT = 4

    def vmfile(filename):
        return os.path.join(vm_dir, filename)

    # 1: has to exist
    if not os.path.isdir(vm_dir):
        return False, "VM dir is not a directory: %s" % (vm_dir)

    # 2: fixed file names exist
    vm = {
        "s": "%s" % (vmfile("vs3dfile.s")),
        "p": "%s" % (vmfile("vp3dfile.p")),
        "d": "%s" % (vmfile("rho3dfile.d")),
    }
    for fixed_name in vm.values():
        if not os.path.exists(fixed_name):
            return False, "VM file not found: %s" % (fixed_name)
    if not os.path.exists(vmfile(VM_PARAMS_FILE_NAME)):
        return False, "VM configuration missing: %s" % (vmfile(VM_PARAMS_FILE_NAME))

    # 3: metadata files exist (made by gen_cords.py)
    vm_conf = load_yaml(vmfile(VM_PARAMS_FILE_NAME))
    meta = {
        "gridfile": "%s" % (vmfile("gridfile%s" % (vm_conf["sufx"]))),
        "gridout": "%s" % (vmfile("gridout%s" % (vm_conf["sufx"]))),
        "bounds": "%s" % (vmfile("model_bounds%s" % (vm_conf["sufx"]))),
        "coords": "%s" % (vmfile("model_coords%s" % (vm_conf["sufx"]))),
        "params": "%s" % (vmfile("model_params%s" % (vm_conf["sufx"]))),
    }
    for meta_file in meta.values():
        if not os.path.exists(meta_file):
            return False, "VM metadata not found: %s" % (meta_file)

    # 4: vm_params.yaml consistency
    try:
        assert vm_conf["nx"] == int(round(vm_conf["extent_x"] / vm_conf["hh"]))
        assert vm_conf["ny"] == int(round(vm_conf["extent_y"] / vm_conf["hh"]))
        assert vm_conf["nz"] == int(
            round((vm_conf["extent_zmax"] - vm_conf["extent_zmin"]) / vm_conf["hh"])
        )
    except AssertionError:
        return (
            False,
            "VM config missmatch between extents and nx, ny, nz: %s"
            % (vmfile(VM_PARAMS_FILE_NAME)),
        )

    # 5: binary file sizes
    vm_size = vm_conf["nx"] * vm_conf["ny"] * vm_conf["nz"] * SIZE_FLOAT
    for bin_file in vm.values():
        size = os.path.getsize(bin_file)
        if size != vm_size:
            return (
                False,
                "VM filesize for %s expected: %d found: %d" % (bin_file, vm_size, size),
            )

    # 6: binary contents
    if numpy:
        # check first zx slice (y = 0)
        smin = np.min(
            np.fromfile(
                vm["s"],
                dtype="<f%d" % (SIZE_FLOAT),
                count=vm_conf["nz"] * vm_conf["nx"],
            )
        )
        pmin = np.min(
            np.fromfile(
                vm["p"],
                dtype="<f%d" % (SIZE_FLOAT),
                count=vm_conf["nz"] * vm_conf["nx"],
            )
        )
        dmin = np.min(
            np.fromfile(
                vm["d"],
                dtype="<f%d" % (SIZE_FLOAT),
                count=vm_conf["nz"] * vm_conf["nx"],
            )
        )
        # works even if min is np.nan
        if not min(smin, pmin, dmin) > 0:
            return False, "VM vs, vp or rho <= 0|nan found: %s" % (vm_dir)

    # 7: contents of meta files
    #    if meta_created:
    #        # TODO: check individual file contents
    #        # not as important, can be re-created based on vm_params.py
    #        pass

    # 8: Check VM within bounds -If DEM file is not present, fails the VM
    if os.path.exists(dem_path):
        with open(dem_path) as dem_fp:
            next(dem_fp)
            lat = next(dem_fp).split()
            min_lat = float(lat[0])
            max_lat = float(lat[-1])
            lon = next(dem_fp).split()
            min_lon = float(lon[0])
            max_lon = float(lon[-1])
        vel_crns_file = os.path.join(vm_dir, "VeloModCorners.txt")
        with open(vel_crns_file) as crns_fp:
            next(crns_fp)
            next(crns_fp)
            for line in crns_fp:
                lon, lat = map(float, line.split())
                if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
                    return False, "VM extents not contained within NZVM DEM"
    else:
        return False, "DEM file missing"

    return True, "VM seems alright: %s." % (vm_dir)


if __name__ == "__main__":
    rc = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("VM_dir", type=str, help="path the VM folder")
    parser.add_argument(
        "-d",
        "--dem_path",
        type=str,
        help="path to the NZVM dem file, "
        "validates that the VM is within the bounds of the DEM",
    )
    args = parser.parse_args()
    try:
        success, message = validate_vm(args.VM_dir, dem_path=args.dem_path)
        if success:
            rc = 0
        else:
            sys.stderr.write("%s\n" % message)
    except Exception as e:
        sys.stderr.write("%s\n" % e)
    finally:
        sys.exit(rc)
