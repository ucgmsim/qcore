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
from qcore.constants import VM_PARAMS_FILE_NAME, VMParams
from qcore.simulation_structure import (
    get_VM_file,
    get_fault_VM_dir,
    verify_VM_files_exist,
)

DEM_PATH = "/nesi/project/nesi00213/opt/Velocity-Model/Data/DEM/NZ_DEM_HD.in"

try:
    import numpy as np

    numpy = True
except ImportError:
    numpy = False


def validate_vm(cybershake_root, fault, dem_path=DEM_PATH):
    """
    Go through rules of VM directories. Return False if invalid.
    cybershake_root: cybershake root folder
    fault: name of the fault the vm is for
    dem_path: path to DEM file
    """
    SIZE_FLOAT = 4

    # 1: has to exist
    vm_dir = get_fault_VM_dir(cybershake_root, fault)
    if not os.path.isdir(vm_dir):
        return False, "VM dir is not a directory: {}".format(vm_dir)
    if not os.path.exists(dem_path):
        return False, "DEM file missing"

    # 2: fixed file names exist
    vm = {"s": "vs3dfile.s", "p": "vp3dfile.p", "d": "rho3dfile.d"}
    result, message = verify_VM_files_exist(cybershake_root, fault, vm.values())
    if not result:
        return result, message
    vm_params = [VM_PARAMS_FILE_NAME]
    result, message = verify_VM_files_exist(cybershake_root, fault, vm_params)
    if not result:
        return result, message
    vm_params_file_path = get_VM_file(cybershake_root, fault, VM_PARAMS_FILE_NAME)

    # 3: metadata files exist (made by gen_cords.py)
    vm_params_dict = load_yaml(vm_params_file_path)
    sufx = vm_params_dict[VMParams.sufx.value]

    meta_files = [
        "gridfile{}".format(sufx),
        "gridout{}".format(sufx),
        "model_bounds{}".format(sufx),
        "model_coords{}".format(sufx),
        "model_params{}".format(sufx),
    ]
    result, message = verify_VM_files_exist(cybershake_root, fault, meta_files)
    if not result:
        return False, message

    # 4: vm_params.yaml consistency
    try:
        assert vm_params_dict[VMParams.nx.value] == int(
            round(
                vm_params_dict[VMParams.extent_x.value]
                / vm_params_dict[VMParams.hh.value]
            )
        )
        assert vm_params_dict[VMParams.ny.value] == int(
            round(
                vm_params_dict[VMParams.extent_y.value]
                / vm_params_dict[VMParams.hh.value]
            )
        )
        assert vm_params_dict[VMParams.nz.value] == int(
            round(
                (
                    vm_params_dict[VMParams.extent_zmax.value]
                    - vm_params_dict[VMParams.extent_zmin.value]
                )
                / vm_params_dict[VMParams.hh.value]
            )
        )
    except AssertionError:
        return (
            False,
            "VM config missmatch between extents and nx, ny, nz: {}".format(
                vm_params_file_path
            ),
        )

    # 5: binary file sizes
    vm_size = (
        vm_params_dict[VMParams.nx.value]
        * vm_params_dict[VMParams.ny.value]
        * vm_params_dict[VMParams.nz.value]
        * SIZE_FLOAT
    )
    for bin_file in vm.values():
        size = os.path.getsize(get_VM_file(cybershake_root, fault, bin_file))
        if size != vm_size:
            return (
                False,
                "VM filesize for {} expected: {} found: {}".format(
                    get_VM_file(cybershake_root, fault, bin_file), vm_size, size
                ),
            )

    # 6: binary contents
    if numpy:
        mins = []
        for file_name in vm.values:
            mins.append(
                np.min(
                    np.fromfile(
                        get_VM_file(cybershake_root, fault, file_name),
                        dtype="<f{}".format(SIZE_FLOAT),
                        count=vm_params_dict[VMParams.nz.value]
                        * vm_params_dict[VMParams.nx.value],
                    )
                )
            )

        # works even if min is np.nan
        if not min(mins) > 0:
            return False, "VM vs, vp or rho <= 0|nan found: {}".format(vm_dir)

    # 7: contents of meta files
    #    if meta_created:
    #        # TODO: check individual file contents
    #        # not as important, can be re-created based on vm_params.py
    #        pass

    # 8: Check VM within bounds -If DEM file is not present, fails the VM
    with open(dem_path) as dem_fp:
        next(dem_fp)
        lat = next(dem_fp).split()
        min_lat = float(lat[0])
        max_lat = float(lat[-1])
        lon = next(dem_fp).split()
        min_lon = float(lon[0])
        max_lon = float(lon[-1])
    vel_crns_file = get_VM_file(cybershake_root, fault, "VeloModCorners.txt")
    with open(vel_crns_file) as crns_fp:
        next(crns_fp)
        next(crns_fp)
        for line in crns_fp:
            lon, lat = map(float, line.split())
            if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
                return False, "VM extents not contained within NZVM DEM"

    return True, "VM seems alright: {}.".format(vm_dir)


if __name__ == "__main__":
    rc = 1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cybershake_root",
        type=str,
        help="The root of the cybershake. Requires the standard cybershake structure.",
    )
    parser.add_argument("fault", type=str, help="The name of the fault")
    parser.add_argument(
        "-d",
        "--dem_path",
        type=str,
        help="path to the NZVM dem file, "
        "validates that the VM is within the bounds of the DEM",
    )
    args = parser.parse_args()
    try:
        success, message = validate_vm(
            args.cybershake_root, args.fault, dem_path=args.dem_path
        )
        if success:
            rc = 0
        else:
            sys.stderr.write("{}\n".format(message))
    except Exception as e:
        sys.stderr.write("{}\n".format(e))
    finally:
        sys.exit(rc)
