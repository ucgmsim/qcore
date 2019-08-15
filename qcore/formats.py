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
        raw_cols = map(str.strip, f.readline().split(","))
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
    dtype[0] = ("station", "|S7")
    dtype[1] = ("component", "|S4")

    # load all at once
    imdb = np.rec.array(
        np.loadtxt(
            csv_file, dtype=dtype, delimiter=",", skiprows=1, usecols=tuple(use_cols)
        )
    )
    if comp is None:
        return imdb
    return imdb[imdb.component == comp]


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
        sep="\s+",
        engine="c",
    )


def load_vs30_file(vs30_file: str):
    """Reads the vs30 file into a pandas dataframe

    :param vs30_file: Path to the vs30 file
    :return: pd.DataFrame
        station as index and columns vs30
    """
    return pd.read_csv(
        vs30_file, sep="\s+", index_col=0, header=None, names=["vs30"]
    )

def load_station_ll_vs30(station_file: str, vs30_file: str):
    """ Reads both station and vs30 file into a single pandas dataframe - keeps only the matching entries

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
