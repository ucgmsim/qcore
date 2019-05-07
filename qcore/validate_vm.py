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
import matplotlib.path as mpltPath
import numpy as np
from qcore.utils import load_yaml
from qcore.constants import VM_PARAMS_FILE_NAME, VMParams
from qcore.srf import get_bounds


DEM_PATH = "/nesi/project/nesi00213/opt/Velocity-Model/Data/DEM/NZ_DEM_HD.in"

try:
    import numpy as np

    numpy = True
except ImportError:
    numpy = False


def validate_vm(vm_dir, dem_path=DEM_PATH, srf=None):
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
        return False, "VM dir is not a directory: {}".format(vm_dir)
    if not os.path.exists(dem_path):
        return False, "DEM file missing"

    # 2: fixed file names exist
    vm = {
        "s": vmfile("vs3dfile.s"),
        "p": vmfile("vp3dfile.p"),
        "d": vmfile("rho3dfile.d"),
    }
    for fixed_name in vm.values():
        if not os.path.exists(fixed_name):
            return False, "VM file not found: {}".format(fixed_name)
    vm_params_file_path = vmfile(VM_PARAMS_FILE_NAME)
    if not os.path.exists(vm_params_file_path):
        return False, "VM configuration missing: {}".format(vm_params_file_path)

    # 3: metadata files exist (made by gen_cords.py)
    vm_params_dict = load_yaml(vm_params_file_path)
    meta = {
        "gridfile": vmfile("gridfile{}".format(vm_params_dict["sufx"])),
        "gridout": vmfile("gridout{}".format(vm_params_dict["sufx"])),
        "bounds": vmfile("model_bounds{}".format(vm_params_dict["sufx"])),
        "coords": vmfile("model_coords{}".format(vm_params_dict["sufx"])),
        "params": vmfile("model_params{}".format(vm_params_dict["sufx"])),
    }
    for meta_file in meta.values():
        if not os.path.exists(meta_file):
            return False, "VM metadata not found: {}".format(meta_file)

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
        size = os.path.getsize(bin_file)
        if size != vm_size:
            return (
                False,
                "VM filesize for {} expected: {} found: {}".format(
                    bin_file, vm_size, size
                ),
            )

    # 6: binary contents
    if numpy:
        mins = []
        for file_name in vm.values():
            mins.append(
                np.min(
                    np.fromfile(
                        file_name,
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
    vel_crns_file = vmfile("VeloModCorners.txt")
    polygon = []
    with open(vel_crns_file) as crns_fp:
        next(crns_fp)
        next(crns_fp)
        for line in crns_fp:
            lon, lat = map(float, line.split())
            polygon.append((lon, lat))
            if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
                return False, "VM extents not contained within NZVM DEM"

    # 9: Check SRF within bounds if given
    if srf is not None:
        srf_bounds = get_bounds(srf)
        edges = []
        for index, start_point in enumerate(polygon):
            end_point = polygon[(index + 1) % len(polygon)]
            lons = np.linspace(start_point[0], end_point[0], 10000)
            lats = compute_intermediate_lat(start_point, end_point, lons)
            edges.extend(list(zip(lons, lats)))

        path = mpltPath.Path(edges)
        for bounds in srf_bounds:
            if not all(path.contains_points(bounds)):
                return False, "Srf extents not contained within velocity model corners"
    return True, "VM seems alright: {}.".format(vm_dir)


def compute_intermediate_lat(lon_lat1, lon_lat2, lon_in):
    conversion_factor = np.pi / 180
    lat1, lon1 = lon_lat1
    lat2, lon2 = lon_lat2
    lat1 *= conversion_factor
    lon1 *= conversion_factor
    lat2 *= conversion_factor
    lon2 *= conversion_factor
    lon = lon_in * conversion_factor
    return (
        np.arctan(
            (
                np.sin(lat1) * np.cos(lat2) * np.sin(lon - lon2)
                - np.sin(lat2) * np.cos(lat1) * np.sin(lon - lon1)
            )
            / (np.cos(lat1) * np.cos(lat2) * np.sin(lon1 - lon2))
        )
    ) / conversion_factor


if __name__ == "__main__":
    rc = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("VM_dir", type=str, help="path the VM folder")
    parser.add_argument(
        "-d",
        "--dem_path",
        default=DEM_PATH,
        type=str,
        help="path to the NZVM dem file, "
        "validates that the VM is within the bounds of the DEM",
    )
    parser.add_argument(
        "-s", "--srf", default=None, type=str, help="An srf related to the VM"
    )
    args = parser.parse_args()
    try:
        success, message = validate_vm(
            args.VM_dir, dem_path=args.dem_path, srf=args.srf
        )
        if success:
            rc = 0
        else:
            sys.stderr.write("{}\n".format(message))
    except Exception as e:
        sys.stderr.write("{}\n".format(e))
    finally:
        sys.exit(rc)
