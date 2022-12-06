"""
Miscellaneous non-specific functions.
Module which contains shared functions/values.

@date 8 April 2016
"""

from __future__ import print_function

import subprocess
import sys
from io import IOBase

# returns a list of stations
# sample line in source file:
#   171.74765   -43.90236 ADCS
def get_stations(source_file, locations=False):
    stations = []
    station_lats = []
    station_lons = []
    with open(source_file, "r") as sp:
        for line in sp.readlines():
            if line[0] not in ["#", "%"]:
                info = line.split()
                stations.append(info[2])
                if locations:
                    station_lons.append(info[0])
                    station_lats.append(info[1])
    if not locations:
        return stations
    return (stations, station_lats, station_lons)


def get_corners(model_params, gmt_format=False):
    """
    Retrieve corners of simulation domain from model params file.
    model_params: file path to model params
    gmt_format: if True, also returns corners in GMT string format
    """
    # with -45 degree rotation:
    #   c2
    # c1  c3
    #   c4
    corners = []
    with open(model_params, "r") as vmpf:
        lines = vmpf.readlines()
        # make sure they are read in the correct order at efficiency cost
        for corner in ["c1=", "c2=", "c3=", "c4="]:
            for line in lines:
                if corner in line:
                    corners.append(list(map(float, line.split()[1:3])))
                    break
    if not gmt_format:
        return corners
    # corners in GMT format
    cnr_str = "\n".join([" ".join(map(str, cnr)) for cnr in corners])
    return corners, cnr_str


def non_blocking_exe(cmd, debug=True, shell=False, stdout=True, stderr=True, **kwargs):
    # always split for consistency
    if type(cmd) == str:
        cmd = cmd.split(" ")

    # display what command would look like if executed on a shell
    if debug:
        virtual_cmd = " ".join(cmd)

        if isinstance(stdout, IOBase):
            virtual_cmd = "%s 1>%s" % (virtual_cmd, stdout.name)
        if isinstance(stderr, IOBase):
            virtual_cmd = "%s 2>%s" % (virtual_cmd, stderr.name)
        print(virtual_cmd, file=sys.stderr)

    # special cases for stderr and stdout
    if stdout == True:
        stdout = subprocess.PIPE
    if stderr == True:
        stderr = subprocess.PIPE

    p = subprocess.Popen(cmd, shell=shell, stdout=stdout, stderr=stderr, **kwargs)
    return p


def exe(cmd, debug=True, shell=False, stdout=True, stderr=True, stdin=None, **kwargs):
    """
    Runs a command in the shell using the provided parameters
    :param cmd: command as list starting with executable, followed by arguments.
         Strings will be split by whitespace even if this splits a parameter.
         This will cause issues when shell == False. List input is ideal.
    :param debug: print equivalent shell command, display output
    :param shell: execute command in shell environment (not recommended)
    :param stdout: True: return output | file: open file object
    :param stderr: True: return error | file: open file object
    :param stdin: None for no input or command input string
    :param non_blocking: True: returns the popen object | False: returns (out, err)
    :param kwargs: Additional arguments to be passed directly to Popen
    :return: the communication object or out/err strings
    """

    p = non_blocking_exe(
        cmd, debug=debug, shell=shell, stdout=stdout, stderr=stderr, **kwargs
    )

    out, err = p.communicate(stdin)
    rc = p.wait()

    if debug:
        if out:
            print(out, file=sys.stderr)
        if err:
            print(err, file=sys.stderr)

    try:
        return out.decode("utf-8"), err.decode("utf-8")
    except:
        return out, err


def is_virtual_station(station_name):
    """
    station_name: (string / unicode)
    Checks if all restraints on virtual station names are met:
    1) Virtual Stations have 7 characters
    2) Virtual Stations contain no capitals
    3) Virtual Stations must be valid hex strings
    """
    # 7 characters long
    if len(station_name) != 7:
        return False

    n_caps = sum(1 for c in station_name if c.isupper())
    if n_caps > 0:
        return False

    # valid hex string
    try:
        if not isinstance(station_name, int):
            int(station_name, 16)
    except (ValueError, TypeError):
        return False

    # all tests passed
    return True
