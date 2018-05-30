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

try:
    import numpy as np
    numpy = True
except ImportError:
    numpy = False

def validate_vm(vm_dir):
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
        return False, 'VM dir is not a directory: %s' % (vm_dir)

    # 2: fixed file names exist
    vm = {'s':'%s' % (vmfile('vs3dfile.s')), \
            'p':'%s' % (vmfile('vp3dfile.p')), \
            'd':'%s' % (vmfile('rho3dfile.d'))}
    for fixed_name in vm.values():
        if not os.path.exists(fixed_name):
            return False, 'VM file not found: %s' % (fixed_name)
    if not os.path.exists(vmfile('params_vel.py')):
        return False, 'VM configuration missing: %s' % (vmfile('params_vel.py'))

    # 3: metadata files exist (made by gen_cords.py)
    sys.path.insert(0, vm_dir)
    import params_vel as vm_conf
    meta = {'gridfile':'%s' % (vmfile('gridfile%s' % (vm_conf.sufx))), \
            'gridout':'%s' % (vmfile('gridout%s' % (vm_conf.sufx))), \
            'bounds':'%s' % (vmfile('model_bounds%s' % (vm_conf.sufx))), \
            'coords':'%s' % (vmfile('model_coords%s' % (vm_conf.sufx))), \
            'params':'%s' % (vmfile('model_params%s' % (vm_conf.sufx)))}
    for meta_file in meta.values():
        if not os.path.exists(meta_file):
            return False, 'VM metadata not found: %s' % (meta_file)

    # 4: params_vel.py consistency
    try:
        nx, ny, nz = map(int, [vm_conf.nx, vm_conf.ny, vm_conf.nz])
        xlen, ylen, zmin, zmax, hh = map(float, \
                [vm_conf.extent_x, vm_conf.extent_y, \
                vm_conf.extent_zmin, vm_conf.extent_zmax, vm_conf.hh])
    except AttributeError:
        return False, 'VM config missing values: %s' % (vmfile('params_vel.py'))
    except ValueError:
        return False, 'VM config contains invalid values: %s' \
                      % (vmfile('params_vel.py'))
    zlen = zmax - zmin
    try:
        assert(nx == int(round(xlen / hh)))
        assert(ny == int(round(ylen / hh)))
        assert(nz == int(round(zlen / hh)))
    except AssertionError:
        return False, 'VM config missmatch between extents and nx, ny, nz: %s' \
                      % (vmfile('params_vel.py'))

    # 5: binary file sizes
    vm_size = nx * ny * nz * SIZE_FLOAT
    for bin_file in vm.values():
        size = os.path.getsize(bin_file)
        if size != vm_size:
            return False, 'VM filesize for %s expected: %d found: %d' \
                    % (bin_file, vm_size, size)

    # 6: binary contents
    if numpy:
        # check first zx slice (y = 0)
        smin = np.min(np.fromfile(vm['s'], dtype = '<f%d' % (SIZE_FLOAT), \
                count = nz * nx))
        pmin = np.min(np.fromfile(vm['p'], dtype = '<f%d' % (SIZE_FLOAT), \
                count = nz * nx))
        dmin = np.min(np.fromfile(vm['d'], dtype = '<f%d' % (SIZE_FLOAT), \
                count = nz * nx))
        # works even if min is np.nan
        if not min(smin, pmin, dmin) > 0:
            return False, 'VM vs, vp or rho <= 0|nan found: %s' % (vm_dir)

    # 7: contents of meta files
    if meta_created:
        # TODO: check individual file contents
        # not as important, can be re-created based on params_vel.py
        pass

    return True, 'VM seems alright: %s.' % (vm_dir))

if __name__ == '__main__':
    rc = 1
    try:
        if len(sys.argv) > 1:
            success, message = validate_vm(sys.argv[1])
            if success:
                rc = 0
            else:
                print(message, file = sys.stderr)
    except Exception as e:
        print(e, file = sys.stderr)
    finally:
        sys.exit(rc)
