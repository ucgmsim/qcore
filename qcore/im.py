"""
Module for handling and analyzing intensity measures (IM) in seismic data.

This module provides functionality for working with intensity measures (IMs), which are used to quantify and analyze seismic activity.
"""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from qcore import constants

DEFAULT_PATTERN_ORDER = (
    "station",
    "component",
    "PGA",
    "PGV",
    "CAV",
    "AI",
    "Ds575",
    "Ds595",
    "Ds2080",
    "MMI",
    "pSA",
    "FAS",
    "SDI",
)


class IMEnum(constants.ExtendedEnum):
    """
    Enum representing different types of intensity measures (IM) used in seismic analysis.

    This enumeration defines a set of standardized intensity measures commonly used to quantify and analyze seismic activity.
    Each member of this enum corresponds to a specific type of intensity measure.

    Members
    -------
    PGA
        Peak Ground Acceleration, a measure of the maximum acceleration of the ground during an earthquake.
    PGV
        Peak Ground Velocity, a measure of the maximum velocity of the ground during an earthquake.
    CAV
        Cumulative Absolute Velocity, the absolute integral of acceleration over time.
    AI
        Arias Intensity, proportional to the integral of the square of ground acceleration over time.
    Ds575
        The duration between 5% and 75% of rupture energy dissapation.
    Ds595
        The duration between 5% and 95% of rupture energy dissapation.
    Ds2080
        The duration between 20% and 80% of rupture energy dissapation.
    MMI
        Modified Mercalli Intensity, a qualitative measure of the effects of an earthquake on people and structures.
    pSA
        Pseudo Spectral Acceleration, a measure of acceleration response at a specific period.
    FAS
        Fourier Amplitude Spectrum, a measure of ground motion as a function of frequency.
    SDI
        Inelastic spectral displacement.
    """

    PGA = enum.auto()
    PGV = enum.auto()
    CAV = enum.auto()
    AI = enum.auto()
    Ds575 = enum.auto()
    Ds595 = enum.auto()
    Ds2080 = enum.auto()
    MMI = enum.auto()
    pSA = enum.auto()
    FAS = enum.auto()
    SDI = enum.auto()


def order_im_cols_file(im_ffp: Union[Path, str]) -> pd.DataFrame:
    """Read an IM file and sort its columns.

    Parameters
    ----------
    im_ffp : Union[Path, str]
        The filepath location of the IM file.

    Returns
    -------
    DataFrame
        A dataframe containing all the IMs with columns in the correct order.
        Correct IM column order:
        station, component, PGA*, PGV*, CAV*, AI*, Ds*, MMI*, pSA_*, FAS_*, SDI_*
    """
    return order_im_cols_df(pd.read_csv(im_ffp))


def order_im_cols_df(
    df: pd.DataFrame, pattern_order: tuple[str, ...] = DEFAULT_PATTERN_ORDER
) -> pd.DataFrame:
    """
    Orders the columns in the dataframe as per the pattern order given.

    All columns that don't match a pattern are just appended to the end in the
    original order.

    Parameters
    ----------
    df : DataFrame
        The dataframe to sort.
    pattern_order : tuple of columns
        The order

    Returns
    -------
    DataFrame
        The dataframe with the columns in the correct order.
    """
    adj_ims = []
    for pattern in pattern_order:
        cur_ims = [im for im in df.columns if im.startswith(pattern)]

        if len(cur_ims) == 0:
            continue
        elif len(cur_ims) == 1:
            adj_ims.append(cur_ims[0])
        else:
            # Check if column name contains a valid float value,
            # e.g. pSA_0.5_epsilon.
            float_ims = []
            for ix, split in enumerate(cur_ims[0].split("_")):
                try:
                    float(split.replace("p", "."))
                    float_ims.append(ix)
                except ValueError:
                    continue

            if len(float_ims) > 0:
                # Get the values (as the list is sorted on those)
                values = []
                for im in cur_ims:
                    values.extend(
                        (
                            list(
                                float(im.split("_")[value_ix].replace("p", "."))
                                for value_ix in float_ims
                            )
                        )
                    )

                sorted_indices = np.argsort(values)

            # Otherwise just sort by length of the column name
            else:
                sorted_indices = np.argsort([len(im) for im in cur_ims])

            # Sort the columns names
            adj_ims.extend(np.asarray(cur_ims)[sorted_indices])
    # Deal with columns that aren't handled by the pattern.
    # These are just added to the end, in the original order
    adj_ims.extend(im for im in df.columns if im not in adj_ims)

    return df[adj_ims]


@dataclass
class IM:
    """Represents an intensity measure (IM) used in seismic analysis.

    This class encapsulates information about an intensity measure, including its name, period, and associated component.
    It provides methods to format the intensity measure's name in various ways and to determine its units.

    Attributes
    ----------
    name : IMEnum
        The name of the intensity measure, represented as an enumeration.
    period : int, optional
        The period associated with the intensity measure, if applicable.
    component : constants.Components, optional
        The component of the seismic signal related to the intensity measure, if specified.

    Methods
    -------
    from_im_name(name: str) -> IM
        Creates an instance of the IM class from a string representation of the intensity measure name.
    """

    name: IMEnum
    period: Optional[int] = None
    component: Optional[constants.Components] = None

    def __post_init__(self):
        """Remap and regularise name and component parameters."""
        if not isinstance(self.name, IMEnum):
            self.name = IMEnum[self.name]

        if self.component is not None and not isinstance(
            self.component, constants.Components
        ):
            self.component = constants.Components.from_str(self.component)

    def get_im_name(self) -> str:
        """str: The qualified IM parameter name."""
        if self.period:
            return f"{self.name}_{self.period}"
        else:
            return f"{self.name}"

    def pretty_im_name(self) -> str:
        """
        str: IM name in the form "IM_NAME [UNITS]" or "IM_NAME (PERIOD|UNITS) [UNITS]"
        """
        if self.period:
            return f"{self.name} ({self.period}{self.get_period_unit()}) [{self.get_unit()}]"
        else:
            return f"{self.name} [{self.get_unit()}]"

    def get_unit(self) -> str:
        """str: The intensity measure unit."""
        if self.name in [IMEnum.PGA, IMEnum.pSA]:
            return "g"
        elif self.name in [IMEnum.PGV, IMEnum.AI]:
            return "cm/s"
        elif self.name in [IMEnum.CAV, IMEnum.FAS]:
            return (
                "gs"  # FAS could also be cm/s depending on calculation / implementation
            )
        elif self.name in [IMEnum.Ds575, IMEnum.Ds595, IMEnum.Ds2080]:
            return "s"
        elif self.name in [IMEnum.MMI]:
            return ""  # MMI is dimensionless
        elif self.name in [IMEnum.SDI]:
            return "cm"
        else:
            return ""  # unimplemented

    def get_period_unit(self):
        """str: The unit of the intensity measure's period."""
        if self.name in [IMEnum.pSA, IMEnum.SDI]:
            return "s"
        elif self.name == IMEnum.FAS:
            return "Hz"
        else:
            return ""

    @staticmethod
    def from_im_name(name: str):
        """Create an IM from a string.

        Parameters
        ----------
        name : str
            The name of the intensity measure (e.g. "PGV")

        Returns
        -------
        IM
            The intensity measure represented by the string.
        """
        parts = name.split("_")
        return IM(*parts)
