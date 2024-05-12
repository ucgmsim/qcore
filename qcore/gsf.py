"""
Module providing utilities for reading GSF files.

Functions:
    read_gsf(gsf_filepath: str) -> pd.DataFrame:
        Parse a GSF file into a pandas DataFrame.
"""

import pandas as pd


def read_gsf(gsf_filepath: str) -> pd.DataFrame:
    """Parse a GSF file into a pandas DataFrame.

    Parameters
    ----------
    gsf_file_handle : TextIO
        The file handle pointing to the GSF file to read.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all the points in the GSF file. The DataFrame's columns are
        - lon (longitude)
        - lat (latitude)
        - depth (Kilometres below ground, i.e. depth = -10 indicates a point 10km underground).
        - sub_dx (The subdivision size in the strike direction)
        - sub_dy (The subdivision size in the dip direction)
        - strike
        - dip
        - rake
        - slip (nearly always -1)
        - init_time (nearly always -1)
        - seg_no (the fault segment this point belongs to)
    """
    with open(gsf_filepath, mode="r", encoding="utf-8") as gsf_file_handle:
        # we could use pd.read_csv with the skiprows argument, but it's not
        # as versatile as simply skipping the first n rows with '#'
        while gsf_file_handle.readline()[0] == "#":
            pass
        # NOTE: This skips the line after the last line beginning with #.
        # This is ok as this line is always the number of points in the GSF
        # file, which we do not need.
        return pd.read_csv(
            gsf_file_handle,
            sep=r"\s+",
            names=[
                "lon",
                "lat",
                "depth",
                "sub_dx",
                "sub_dy",
                "strike",
                "dip",
                "rake",
                "slip",
                "init_time",
                "seg_no",
            ],
        )
