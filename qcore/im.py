"""
Correct IM column order
station, component, PGA*, PGV*, CAV*, AI*, Ds*, MMI*, pSA_*
"""

import pandas as pd
import numpy as np

default_pattern_order = ("station", "component", "PGA", "PGV", "CAV", "AI",
                         "Ds575", "Ds595", "Ds2080", "MMI", "pSA")


def order_im_cols_file(filename):
    """For a full description see function order_im_cols_df"""
    df = pd.read_csv(filename)

    return order_im_cols_df(df)


def order_im_cols_df(df, pattern_order=default_pattern_order):
    """Orders the columns in the dataframe as per the pattern order given.

    If there are several columns matching a pattern, and the column contains a
    number seperated by a '_', such as pSA_0.5, then those columns are sorted
    lowest to highest based on the number. The number has to be in the same
    position for all column names of a pattern.

    All columns that don't match a pattern are just appended to the end in the
    original order.
    """
    orig_cols = df.columns
    adj_cols = []

    for pattern in pattern_order:
        cur_cols = [col for col in orig_cols if col.startswith(pattern)]

        if len(cur_cols) == 0:
            continue
        elif len(cur_cols) == 1:
            adj_cols.append(cur_cols[0])
        else:
            # Check if column name contains a valid float value,
            # e.g. pSA_0.5_epsilon.
            contains_float, value_ix = False, None
            for ix, split in enumerate(cur_cols[0].split("_")):
                try:
                    float(split.replace("p", "."))
                    contains_float, value_ix = True, ix
                    break
                except ValueError:
                    continue

            if contains_float:
                # Get the values (as the list is sorted on those)
                values = [float(col.split("_")[value_ix].replace("p", "."))
                          for col in cur_cols]

                sorted_indices = np.argsort(values)

            # Otherwise just sort by length of the column name
            else:
                sorted_indices = np.argsort([len(col) for col in cur_cols])

            # Sort the columns names
            adj_cols = adj_cols \
                + list(np.asarray(cur_cols)[sorted_indices])

    # Deal with columns that aren't handled by the pattern.
    # These are just added to the end, in the original order
    if len(adj_cols) != len(orig_cols):
        [adj_cols.append(col) for col in orig_cols if col not in adj_cols]

    return df[adj_cols]





