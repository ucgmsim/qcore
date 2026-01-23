"""
Functions and classes to load data that doesn't belong elsewhere.
"""

import pandas as pd
import numpy as np


def load_im_file(csv_file, all_psa=False, comp=None):

    # process column names
    use_cols = []
    col_names = []
    with open(csv_file, "r") as f:
        raw_cols = list(map(str.strip, f.readline().split(",")))
    for i, c in enumerate(raw_cols):
        # filter out pSA that aren't round numbers, duplicates
        if c not in col_names and (
            all_psa or not (c.startswith("pSA_") and len(c) > 12)
        ):
            use_cols.append(i)
            col_names.append(c)

    # create numpy datatype
    dtype = [(n, np.float32) for n in col_names]
    # first 2 columns are actually strings
    # Non uniform grid station names are a maximum of 7 chars (EMOD restriction)
    # Component has been set to 10 to accomodate ROTD100_50
    dtype[0] = ("station", "|U7")
    dtype[1] = ("component", "|U10")

    # load all at once
    imdb = np.rec.array(
        np.loadtxt(
            csv_file, dtype=dtype, delimiter=",", skiprows=1, usecols=tuple(use_cols)
        )
    )
    if comp is None:
        return imdb
    return imdb[imdb.component == comp]


def load_im_file_pd(imcsv, all_ims=False, comp=None):
    """
    Loads an IM file using pandas and returns a dataframe
    :param imcsv: FFP to im_csv
    :param all_ims: returns all_ims. Defaultly returns only short IM names (standard pSA periods etc).
                    Setting this to true includes all pSA periods (and other long IM names). Extended pSA periods have
                    longer IM names and are filtered out by this flag.
    :param comp: component to return. Default is to return all
    :return:
    """
    df = pd.read_csv(imcsv, index_col=[0, 1])

    if not all_ims:
        df = df[[im for im in df.columns if (len(im) < 15)]]

    if comp is not None:
        df = df[df.index.get_level_values(1) == comp]

    return df


def load_station_file(station_file: str):
    """Reads the station file into a pandas dataframe

    Parameters
    ----------
    station_file : str
        Path to the station file

    Returns
    -------
    pd.DataFrame
        station as index and columns lon, lat
    """
    return pd.read_csv(
        station_file,
        header=None,
        index_col=2,
        names=["lon", "lat"],
        engine="c",
        delim_whitespace=True,
    )


def load_vs30_file(vs30_file: str):
    """Reads the vs30 file into a pandas dataframe

    :param vs30_file: Path to the vs30 file
    :return: pd.DataFrame
        station as index and columns vs30
    """
    return pd.read_csv(vs30_file, sep="\s+", index_col=0, header=None, names=["vs30"])


def load_z_file(z_file: str):
    """Reads the z file into a pandas dataframe

    :param z_file: Path to the z file
    :return: pd.DataFrame
        station as index and columns z1p0, z2p5
    """
    return pd.read_csv(z_file, names=["z1p0", "z2p5", "sigma"], index_col=0, skiprows=1)


def load_station_ll_vs30(station_file: str, vs30_file: str):
    """Reads both station and vs30 file into a single pandas dataframe - keeps only the matching entries

    :param station_file: Path to the station file
    :param vs30_file: Path to the vs30 file
    :return: pd.DataFrame
        station as index and columns lon, lat, vs30
    """

    vs30_df = load_vs30_file(vs30_file)
    station_df = load_station_file(station_file)

    return vs30_df.merge(station_df, left_index=True, right_index=True)


def load_rrup_file(rrup_file: str):
    """Reads the rrup file into a pandas dataframe

    Parameters
    ----------
    rrup_file: str
        Path to the rrup file to load

    Returns
    -------
    pd.DataFrame
        station as index with columns rrup, rjb and optional rx
    """
    return pd.read_csv(rrup_file, header=0, index_col=0, engine="c")


def load_fault_selection_file(fault_selection_file):
    """
    Loads a fault selection file, returning a dictionary of fault:count pairs
    :param fault_selection_file: The relative or absolute path to the fault selection file
    :return: A dictionary of fault:count pairs for all faults found in the file
    """
    faults = {}
    with open(fault_selection_file) as fault_file:
        for lineno, line in enumerate(fault_file.readlines()):
            if len(line) == 0 or len(line.lstrip()) == 0 or line.lstrip()[0] == "#":
                # Line is either empty only whitespace or commented out
                continue
            try:
                line_parts = line.split()
                fault = line_parts[0]
                if len(line_parts) == 1:
                    count = 0
                elif len(line_parts) == 2:
                    count = line_parts[1]
                    if count.endswith("r"):
                        count = int(count[:-1])
                    else:
                        count = int(count)
                else:
                    raise ValueError()
            except ValueError:
                raise ValueError(
                    "Error encountered on line {lineno} when loading fault selection file {fault_selection_file}. "
                    "Line content: {line}".format(
                        lineno=lineno,
                        fault_selection_file=fault_selection_file,
                        line=line,
                    )
                )
            if fault in faults.keys():
                raise ValueError(
                    "Fault {} has been found twice in the fault selection file, please check the file".format(
                        fault
                    )
                )
            faults.update({fault: count})

    return faults


def load_e3d_par(fp: str, comment_chars=("#",)):
    """
    Load an emod3d parameter file as a dictionary.

    Parses key=value pairs from an emod3d parameter file. Since the original
    file format does not include type information, all values are returned as
    strings. Blank lines and comment lines are skipped.

    Parameters
    ----------
    fp : str
        The path to the parameter file.
    comment_chars : tuple of str, optional
        Characters that denote a comment line when they are the first
        non-whitespace character. Default is ("#",).

    Returns
    -------
    dict
        Dictionary of key:value pairs as found in the parameter file.
        All values are strings.

    Raises
    ------
    ValueError
        If a line cannot be parsed (e.g., missing "=" separator).

    Notes
    -----
    Changes by Andrew Ridden-Harper on 23 Jan 2026:

    - Added handling for blank lines (lines containing only whitespace are now
      skipped instead of causing a crash).
    - Improved error messages to show the problematic line when parsing fails.
    - Disabled the duplicate key check. Some e3d.par files in Cybershake v25p11
      have duplicate keys, where the second occurrence is in a section called
      "--- Run-Specific Overrides ---", intended to override default values.
      Later values now override earlier ones. For a specific example, see
      e3d.par for MS09/MS09_REL01.
    """
    vals = {}
    with open(fp) as e3d:
        for line in e3d:
            try:
                # Changed by Andrew Ridden-Harper on 23 Jan 2026 to skip blank lines (to only contain a new line character)
                # and show the line that caused the crash in the error message.
                stripped_line = line.strip()
                if (len(stripped_line) == 0) or (stripped_line[0] in comment_chars):
                    continue

                key, value = stripped_line.split("=")
            except:
                raise ValueError(f"Crashed trying to parse the following line: {line}")

            # Andrew Ridden-Harper disabled this check on 23 Jan 2026 because some of the e3d.par files
            # in Cybershake v25p11 have duplicate keys, where the second key is in a section called
            # " --- Run-Specific Overrides --- " which suggests that they are intended to override the default values.
            # For a specific example, see e3d.par for MS09/MS09_REL01.
            #
            # if key in vals:
            #     raise KeyError(
            #         f"Key {key} is in the emod3d parameter file at least twice. Resolve this before re running."
            #     )
            vals[key] = value

    return vals


