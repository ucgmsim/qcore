"""
Correct IM column order
station, component, PGA*, PGV*, CAV*, AI*, Ds*, MMI*, pSA_*, FAS_*, IESDR_*
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np

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
    "IESDR",
)


class IMEnum(constants.ExtendedEnum):
    PGA = 1
    PGV = 2
    CAV = 3
    AI = 4
    Ds575 = 5
    Ds595 = 6
    Ds2080 = 7
    MMI = 8
    pSA = 9
    FAS = 10
    IESDR = 11


def order_im_cols_file(filename):
    """
    For a full description see function order_im_cols_df
    """
    df = pd.read_csv(filename)

    return order_im_cols_df(df)


def order_im_cols_df(df, pattern_order=DEFAULT_PATTERN_ORDER):
    """
    Orders the columns in the dataframe as per the pattern order given.

    All columns that don't match a pattern are just appended to the end in the
    original order.
    """
    adj_cols = order_ims(df.columns, pattern_order=pattern_order)

    return df[adj_cols]


def order_ims(unsorted_ims, pattern_order=DEFAULT_PATTERN_ORDER):
    """
    Orders the ims as per the pattern order given.

    If there are several columns matching a pattern, and the column contains a
    number seperated by a '_', such as pSA_0.5, then those columns are sorted
    lowest to highest based on the number. The number has to be in the same
    position for all column names of a pattern.
    """
    adj_ims = []
    for pattern in pattern_order:
        cur_ims = [im for im in unsorted_ims if im.startswith(pattern)]

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
            adj_ims = adj_ims + list(np.asarray(cur_ims)[sorted_indices])
    # Deal with columns that aren't handled by the pattern.
    # These are just added to the end, in the original order
    if len(adj_ims) != len(unsorted_ims):
        [adj_ims.append(im) for im in unsorted_ims if im not in adj_ims]

    return adj_ims


@dataclass
class IM:
    name: [IMEnum] = None
    period: [int] = None
    component: [constants.Components] = None

    def get_im_name(self):
        if self.period:
            return f"{self.name}_{self.period}"
        else:
            return self.name

    def pretty_im_name(self):
        if self.period:
            return f"{self.name} ({self.period}{self.get_period_unit()}) [{self.get_unit()}]"
        else:
            return f"{self.name} [{self.get_unit()}]"

    def get_unit(self):
        if self.name in ["PGA", "pSA"]:
            return "g"
        elif self.name == "PGV":
            return "cm/s"
        elif self.name == "CAV":
            return "gs"
        elif self.name == "AI":
            return "cm/s"
        elif self.name == "MMI":
            return ""  # MMI is dimensionless
        elif self.name == "FAS":
            return "gs"
        elif self.name == "IESDR":
            return ""
        elif self.name.startswith("Ds"):
            return "s"
        else:
            return ""  # unimplemented

    def get_period_unit(self):
        if self.name in ["pSA", "IESDR"]:
            return "s"
        elif self.name == "FAS":
            return "Hz"
