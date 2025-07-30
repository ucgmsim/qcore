"""Testing for qcore.siteamp module"""

from pathlib import Path

import pandas as pd
import pytest

from qcore import siteamp_models


@pytest.fixture(scope="module")
def cb_2014_df() -> pd.DataFrame:
    """CB2014 Siteamp Dataframe.

    Values are site-amplification factors derived from CB2014 spreadsheet made by Felipe Kuncar.
    Spreadsheet link: https://www.dropbox.com/scl/fi/spbolh0iiy57pqlv9fx6h/CB14old.xlsx?rlkey=wtnqljwcwo1uhtczf1vz1771p&st=7wz5l89a&dl=0
    Spreadsheet dropbox path: /QuakeCoRE/Public/Test

    Returns
    -------
    pd.DataFrame
        CB2014 Siteamp reference database.
    """
    return pd.read_csv(Path(__file__).parent / "cb_2014_test.csv")


def test_cb_2014_siteamp_model(cb_2014_df: pd.DataFrame) -> None:
    """Test CB2014 values against reference implementation.

    Parameters
    ----------
    cb_2014_df : pd.DataFrame
        The dataframe containing the CB2014 reference dataset.
    """
    input_df = pd.DataFrame(
        {
            "vref": 500.0,
            "vpga": 500.0,
            "pga": cb_2014_df["PGA"],
            "vsite": cb_2014_df["Vs30"],
        }
    )
    # The first column value is 1000Hz but is not in the test dataset, so drop it with 1:
    output = siteamp_models.cb_amp_multi(input_df, 0, 0)[:, 1:]
    # Dataframe has PGA and Vs30 columns, drop those as they are inputs.
    expected = cb_2014_df.to_numpy()[:, 2:]
    # Test equality to within a 1% tolerance
    assert output == pytest.approx(expected, rel=0.01, abs=0.0)
