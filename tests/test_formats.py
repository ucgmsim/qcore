"""Tests for qcore.formats."""

import warnings
from pathlib import Path

from qcore import formats


def test_load_generic_station_file_non_default_name_col(tmp_path: Path) -> None:
    """Station name column not at its rank among selected columns is the index."""
    stat_file = tmp_path / "stations.ll"
    stat_file.write_text("171.1 -43.2 x ADCS y basinA\n172.1 -42.2 x BBBB y basinB\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = formats.load_generic_station_file(
            str(stat_file), stat_name_col=3, other_cols=[5], other_names=["basin"]
        )
    assert df.index.name == "stat_name"
    assert list(df.index) == ["ADCS", "BBBB"]
    assert list(df["basin"]) == ["basinA", "basinB"]
    assert list(df["lon"]) == [171.1, 172.1]


def test_load_generic_station_file_name_col_beyond_selected(tmp_path: Path) -> None:
    """Station name column number exceeding the number of selected columns works."""
    stat_file = tmp_path / "stations.ll"
    stat_file.write_text("171.1 -43.2 a b c ADCS\n172.1 -42.2 a b c BBBB\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = formats.load_generic_station_file(str(stat_file), stat_name_col=5)
    assert list(df.index) == ["ADCS", "BBBB"]
    assert list(df["lat"]) == [-43.2, -42.2]
