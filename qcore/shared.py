"""
Miscellaneous functions whose usage is shared across a number of repositories.
"""

import re
import subprocess
import sys
from io import FileIO
from pathlib import Path
from typing import AnyStr, Optional, Union
from warnings import deprecated

import pandas as pd


def get_stations(
    station_ffp: Union[Path, str], locations: bool = False
) -> Union[list[str], tuple[list[str], list[str], list[str]]]:
    """Parse a station list file.

    Sample line in source file:
      171.74765   -43.90236 ADCS

    Parameters
    ----------
    station_ffp : A Path or str
        The file path of the station file.
    locations : bool, default False
        If True, also return locations.

    Returns
    -------
    stations : list[str]
        A list of station names in the file.
    latitudes : list[float], if locations is True
        The latitudes of the stations.
    longitudes : list[float], if locations is True
        The longitudes of the stations.
    """
    stations = pd.read_csv(
        station_ffp,
        comment="#",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "station"],
        skipinitialspace=True,
    )

    if not locations:
        return stations["station"].to_list()
    return (
        stations["station"].to_list(),
        stations["latitude"].to_list(),
        stations["longitude"].to_list(),
    )


def get_corners(
    model_params_ffp: Union[Path, str], gmt_format: bool = False
) -> Union[list[tuple[float, float]], tuple[list[tuple[float, float]], str]]:
    """
    Retrieve corners of simulation domain from model params file.

    Parameters
    ----------
    model_params_ffp : Path or str
        The file path of the model_params file.
    gmt_format : bool, default False
        If True, also returns corners in GMT string format.

    Returns
    -------
    list[tuple[float, float]]
        A list of (lon, lat) pairs read from the model params file.
    str if gmt_format is True
        A gmt format string specifying the corners read from the model params file
    """
    # with -45 degree rotation:
    #   c2
    # c1  c3
    #   c4
    with open(model_params_ffp, "r") as model_params_file_handle:
        corners = {}
        for line in model_params_file_handle:
            # Matching a string "c{i}= {lon} {lat}", e.g. c1= 172.16403 -41.41359
            corner_match = re.match(
                r"\s+c(\d)=\s+([0-9\.\-\+]+)\s+([0-9\.\-\+]+)", line
            )
            if corner_match:
                corners[int(corner_match.group(1)) - 1] = (
                    float(corner_match.group(2)),
                    float(corner_match.group(3)),
                )
        corners = [corners[i] for i in range(4)]

    if not gmt_format:
        return corners
    # corners in GMT format
    cnr_str = "\n".join(f"{lon} {lat}" for lon, lat in corners)
    return corners, cnr_str


@deprecated("use subprocess.run or subprocess.check_call")
def non_blocking_exe(
    cmd: Union[str, list[str]],
    debug: bool = True,
    stdout: Union[bool, FileIO] = True,
    stderr: Union[bool, FileIO] = True,
    **kwargs,
) -> subprocess.Popen:  # pragma: no cover
    r"""Run a command without blocking the calling thread.

    *DO NOT USE THIS FUNCTION* Instead, call subprocess.run or
    subprocess.check_call to execute processes.

    >>> non_blocking_exe('ls')
    # Some popen object

    Becomes

    >>> subprocess.check_output(['ls']).decode('utf-8')
    "file1.py\nfile2.py\n..."

    Parameters
    ----------
    cmd : str or list[str]
        The command to execute, or a list strings to use the command argument
        to Popen.
    debug : bool, default True
        If True, print out the command to run before running.
    stdout : bool or FileIO, default True
        The stdout file handle to send output. If True, will default to
        subprocess.PIPE.
    stderr : bool or FileIO, default True
        The stderr file handle to send output. If True, will default to
        subprocess.PIPE.
    **kwargs : dict
        Additional arguments, passed to Popen.

    Returns
    -------
    Popen
        The process executed by the function.
    """
    # always split for consistency
    if isinstance(cmd, str):
        cmd = cmd.split(" ")

    # display what command would look like if executed on a shell
    if debug:
        virtual_cmd = " ".join(cmd)

        if isinstance(stdout, FileIO):
            virtual_cmd += f" 1>{stdout.name}"
        if isinstance(stderr, FileIO):
            virtual_cmd += f" 2>{stderr.name}"
        print(virtual_cmd, file=sys.stderr)

    # special cases for stderr and stdout
    stdout_pipe = stdout
    stderr_pipe = stderr
    if stdout is True:
        stdout_pipe = subprocess.PIPE
    if stderr is True:
        stderr_pipe = subprocess.PIPE

    p = subprocess.Popen(cmd, stdout=stdout_pipe, stderr=stderr_pipe, **kwargs)
    return p


@deprecated("use subprocess.run or subprocess.check_call")
def exe(
    cmd: Union[str, list[str]],
    debug: bool = True,
    stdin: Optional[AnyStr] = None,
    **kwargs,
) -> Union[tuple[str, str], tuple[bytes, bytes]]:  # pragma: no cover
    """
    Runs a command in the shell using the provided parameters.

    *DO NOT USE THIS FUNCTION* Instead, call subprocess.run or
    subprocess.check_call to execute processes.

    Parameters
    ----------
    cmd : str or list[str]
        Command as list starting with executable, followed by arguments.
        Strings will be split by whitespace even if this splits a parameter.
        This will cause issues when shell == False. List input is ideal.
    debug : bool, default True
        If True, print out the command to run before running.
    stdin : str-like, optional
        If not None, then given to the running process as standard input.
    **kwargs : dict
        Additional arguments, passed to Popen.

    Returns
    -------
    stdout : str or bytes
        The utf-8 decoded output of the command, or bytes if utf-8 conversion
        fails.
    stdout : str or bytes
        The utf-8 decoded error output of the command, or bytes if utf-8
        conversion fails.
    """

    exe_process = non_blocking_exe(cmd, debug=debug, **kwargs)  # type: ignore

    out, err = exe_process.communicate(stdin)
    _ = exe_process.wait()

    if debug:
        if out:
            print(out, file=sys.stderr)
        if err:
            print(err, file=sys.stderr)

    try:
        return out.decode("utf-8"), err.decode("utf-8")
    except UnicodeDecodeError:
        return out, err


def is_virtual_station(station_name: str) -> bool:
    """Check if the given station identifier is a virtual station.

    1) Virtual Stations have 7 characters
    2) Virtual Stations contain no capitals
    3) Virtual Stations must be valid hex strings

    Parameters
    ----------
    station_name : str
        The name of the station.

    Returns
    -------
    bool
        True if the station name is a virtual station identifier.
    """
    return bool(re.match(r"[0-9a-f]{7}", station_name))
