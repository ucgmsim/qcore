"""
Functions and classes to load data that doesn't belong elsewhere.
"""

import argparse
from pathlib import Path
from typing import overload

import pandas as pd
from typing_extensions import deprecated


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_im_file_pd(
    imcsv: Path | str, all_ims: bool = False, comp: str | None = None
) -> pd.DataFrame | pd.Series:
    """Load an Intensity Measure (IM) CSV file into a pandas DataFrame.

    Parameters
    ----------
    imcsv : Path or str
        Path to the IM CSV file.
    all_ims : bool, optional
        Whether to load all IMs. If False (default), only standard IMs
        with short names (e.g., common pSA periods) are included.
    comp : str or None, optional
        Specific component to return (e.g., 'pga', 'pgv'). If None, all components are returned.

    Returns
    -------
    pd.DataFrame or series
        DataFrame or series containing IM values, indexed by station and component.
    """
    df = pd.read_csv(imcsv, index_col=[0, 1])

    if not all_ims:
        df = df[[im for im in df.columns if (len(im) < 15)]]

    if comp is not None:
        df = df[df.index.get_level_values(1) == comp]

    return df


@overload
def station_file_argparser(
    parser: argparse.ArgumentParser,
) -> None: ...  # numpydoc ignore=GL08


@overload
def station_file_argparser() -> argparse.ArgumentParser: ...  # numpydoc ignore=GL08


@deprecated("Will be removed after Cybershake investigation concludes.")
def station_file_argparser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser | None:
    """Add station file argument options to an ArgumentParser.

    Parameters
    ----------
    parser : argparse.ArgumentParser or None, optional
        Existing parser to extend. If None, a new parser is created.

    Returns
    -------
    argparse.ArgumentParser
        Parser object with added station file-related arguments.

    Examples
    --------
    >>> parser = station_file_argparser()
    >>> args = parser.parse_args(["--stat_file", "stations.ll"])
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Station Data Loader")
    arg = parser.add_argument

    arg(
        "--stat_file",
        help="add longitude latitude station file to plot (eg. *.ll file)",
    )
    arg("--sep", help="delimiter", default=" ")
    arg("--skiprows", help="number of rows to skip", type=int, default=0)
    arg(
        "--stat_name_col",
        help="the column number of the station name to be used as index",
        type=int,
        default=2,
    )
    arg(
        "--lon_col",
        help="Column number to be used as the longitude",
        type=int,
        default=0,
    )
    arg(
        "--lat_col",
        help="Column number to be used as the latitude",
        type=int,
        default=1,
    )
    arg(
        "--other_cols",
        help="Other columns to include. eg. --other_cols 3 4",
        nargs="+",
        type=int,
        default=[],
    )
    arg(
        "--other_names",
        help="Other column names to include. eg. --other_names basin basin_type",
        nargs="+",
        default=[],
    )

    return parser


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_generic_station_file(
    stat_file: str,
    stat_name_col: int = 2,
    lon_col: int = 0,
    lat_col: int = 1,
    other_cols: list[int] | None = None,
    other_names: list[str] | None = None,
    sep: str = r"\s+",
    skiprows: int = 0,
) -> pd.DataFrame:
    """Load a generic station file into a pandas DataFrame.

    Parameters
    ----------
    stat_file : str
        Path to the station file (e.g., .ll file or other format).
    stat_name_col : int, optional
        Column index for station names. Default is 2.
    lon_col : int, optional
        Column index for longitude. Default is 0.
    lat_col : int, optional
        Column index for latitude. Default is 1.
    other_cols : list of int, optional
        Indices of additional columns to include.
    other_names : list of str, optional
        Names corresponding to `other_cols`.
    sep : str, optional
        Column delimiter. Default is whitespace.
    skiprows : int, optional
        Number of rows to skip (for header lines).

    Returns
    -------
    pd.DataFrame
        DataFrame with index as station name and columns including longitude,
        latitude, and any additional specified columns.
    """
    cols: dict[str, int] = {"stat_name": stat_name_col}
    if lon_col is not None:
        cols["lon"] = lon_col
    if lat_col is not None:
        cols["lat"] = lat_col

    if other_cols and other_names:
        for col_idx, col_name in zip(other_cols, other_names):
            cols[col_name] = col_idx
    return pd.read_csv(
        stat_file,
        # we will be loading columns of these indices (order doesn't matter)
        usecols=list(cols.values()),
        names=sorted(
            cols, key=cols.get
        ),  # eg. cols={stat_name:2, lon:0, lat:1} means names = ["lon","lat","stat_name"] # type: ignore[no-matching-overload]
        index_col=stat_name_col,
        sep=sep,
        header=None,
        skiprows=skiprows,
    )  # type: ignore


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_station_file(station_file: str) -> pd.DataFrame:
    """Load a station file into a pandas DataFrame.

    Parameters
    ----------
    station_file : str
        Path to the station file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by station, with longitude and latitude columns.
    """
    return pd.read_csv(
        station_file,
        header=None,
        index_col=2,
        names=["lon", "lat"],
        engine="c",
        delim_whitespace=True,
    )  # type: ignore[no-matching-overload]


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_vs30_file(vs30_file: str) -> pd.DataFrame:
    """Load a Vs30 (shear-wave velocity) file into a pandas DataFrame.

    Parameters
    ----------
    vs30_file : str
        Path to the Vs30 file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by station, with a single column ``vs30``.
    """
    return pd.read_csv(vs30_file, sep=r"\s+", index_col=0, header=None, names=["vs30"])


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_z_file(z_file: str) -> pd.DataFrame:
    """Load a z-file containing depth parameters (e.g., z1.0, z2.5) into a pandas DataFrame.

    Parameters
    ----------
    z_file : str
        Path to the z file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by station, with columns ``z1p0``, ``z2p5``, and ``sigma``.
    """
    return pd.read_csv(z_file, names=["z1p0", "z2p5", "sigma"], index_col=0, skiprows=1)


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_station_ll_vs30(station_file: str, vs30_file: str) -> pd.DataFrame:
    """Merge station location and Vs30 data into a single DataFrame.

    Parameters
    ----------
    station_file : str
        Path to the station file containing longitude and latitude.
    vs30_file : str
        Path to the Vs30 file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by station, with columns ``lon``, ``lat``, and ``vs30``.
    """

    vs30_df = load_vs30_file(vs30_file)  # type: ignore
    station_df = load_station_file(station_file)  # type: ignore

    return vs30_df.merge(station_df, left_index=True, right_index=True)


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_rrup_file(rrup_file: str) -> pd.DataFrame:
    """Reads the rrup file into a pandas dataframe

    Parameters
    ----------
    rrup_file : str
        Path to the rrup file to load

    Returns
    -------
    pd.DataFrame
        Station as index with columns rrup, rjb and optional rx
    """
    return pd.read_csv(rrup_file, header=0, index_col=0, engine="c")


@deprecated("Will be removed after Cybershake investigation concludes.")
def load_fault_selection_file(fault_selection_file: str | Path) -> dict[str, int]:
    """Load a fault selection file into a dictionary of fault names and counts.

    Parameters
    ----------
    fault_selection_file : str or Path
        Path to the fault selection file.

    Returns
    -------
    dict of str to int
        Dictionary mapping fault names to selected counts.

    Raises
    ------
    ValueError
        If the file contains malformed lines or duplicate fault entries.
    """
    faults: dict[str, int] = {}
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
                    f"Error encountered on line {lineno} when loading fault selection file {fault_selection_file}. "
                    f"Line content: {line}"
                )
            if fault in faults.keys():
                raise ValueError(
                    f"Fault {fault} has been found twice in the fault selection file, please check the file"
                )
            faults.update({fault: count})

    return faults
