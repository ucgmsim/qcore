#!/usr/bin/env python2

import os
from subprocess import check_call
import sys

from qcore.binary_version import get_unversioned_bin
from qcore.utils import load_yaml

def gen_coords(vm_dir=".", debug=False, geoproj="1", do_coords="1", centre_origin="1"):
    """
    Generate coordinate files for an emod3d domain (set of grid points).
    outdir: directory to store coordinate files in
    debug: print additional info
    geoproj: 
    do_coords: 
    """

    # load params for velocity model
    try:
        vm=load_yaml(os.path.join(vm_dir,"params_vel.yaml"))
    except (IOError, FileNotFoundError) as e:
        # deprecated, will break if run multiple times as a function
        sys.path.insert(0, vm_dir)
        import params_vel as vm

        # notify of location because of above breakage issue
        print("vm dir:     %s\nsim params: %s" % (vm_dir, os.path.abspath(vm.__file__)))
        vm = {
            "nx": int(vm.nx),
            "ny": int(vm.ny),
            "nz": int(vm.nz),
            "hh": float(vm.hh),
            "sufx": vm.sufx,
            "MODEL_LON": float(vm.MODEL_LON),
            "MODEL_LAT": float(vm.MODEL_LAT),
            "MODEL_ROT": float(vm.MODEL_ROT),
        }
    XLEN = vm["nx"] * vm["hh"]
    YLEN = vm["ny"] * vm["hh"]
    ZLEN = vm["nz"] * vm["hh"]

    # list of outputs that this function can create
    GRIDFILE = os.path.join(vm_dir, "gridfile%s" % (vm["sufx"]))
    GRIDOUT = os.path.join(vm_dir, "gridout%s" % (vm["sufx"]))
    MODEL_COORDS = os.path.join(vm_dir, "model_coords%s" % (vm["sufx"]))
    MODEL_PARAMS = os.path.join(vm_dir, "model_params%s" % (vm["sufx"]))
    MODEL_BOUNDS = os.path.join(vm_dir, "model_bounds%s" % (vm["sufx"]))

    # generate gridfile
    try:
        with open(GRIDFILE, "w") as gridf:
            gridf.write("xlen=%f\n" % (XLEN))
            gridf.write("%10.4f %10.4f %13.6e\n" % (0.0, XLEN, vm["hh"]))
            gridf.write("ylen=%f\n" % (YLEN))
            gridf.write("%10.4f %10.4f %13.6e\n" % (0.0, YLEN, vm["hh"]))
            gridf.write("zlen=%f\n" % (ZLEN))
            gridf.write("%10.4f %10.4f %13.6e\n" % (0.0, ZLEN, vm["hh"]))
    except IOError:
        raise IOError("Cannot write GRIDFILE: %s" % (GRIDFILE))

    # generate model_params
    cmd = (
        "{} "
        "geoproj={geoproj} gridfile='{GRIDFILE}' gridout='{GRIDOUT}' "
        "center_origin={centre_origin} do_coords={do_coords} "
        "nzout=1 name='{MODEL_COORDS}' gzip=0 latfirst=0 "
        "modellon={vm[MODEL_LON]} modellat={vm[MODEL_LAT]} "
        "modelrot={vm[MODEL_ROT]} 1> '{MODEL_PARAMS}'"
    ).format(get_unversioned_bin("gen_model_cords"), **dict(locals(), **globals()))
    if debug:
        print(cmd)
    else:
        cmd += " 2>/dev/null"
    check_call(cmd, shell=True)

    # also generate coordinate related outputs
    if do_coords != "1":
        return

    # retrieve MODEL_BOUNDS
    x_bounds = [0, vm["nx"] - 1]
    y_bounds = [0, vm["ny"] - 1]
    try:
        with open(MODEL_COORDS, "r") as coordf:
            with open(MODEL_BOUNDS, "w") as boundf:
                for line in coordf:
                    x, y = map(float, line.split()[2:4])
                    if x in x_bounds or y in y_bounds:
                        boundf.write(line)
    except IOError:
        raise IOError("Cannot write MODEL_BOUNDS: %s" % (MODEL_BOUNDS))


# allow running from shell
if __name__ == "__main__":
    if len(sys.argv) > 1:
        gen_coords(vm_dir=sys.argv[1])
    else:
        gen_coords()
