"""Tests for qcore.formats."""

from pathlib import Path

import pandas as pd
import pytest

from qcore import formats


def test_load_generic_station_file_non_default_name_col(tmp_path: Path) -> None:
    """Station name column not at its rank among selected columns is the index."""
    stat_file = tmp_path / "stations.ll"
    stat_file.write_text("171.1 -43.2 x ADCS y basinA\n172.1 -42.2 x BBBB y basinB\n")
    with pytest.deprecated_call():
        df = formats.load_generic_station_file(  # ty: ignore[deprecated]
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
    with pytest.deprecated_call():
        df = formats.load_generic_station_file(  # ty: ignore[deprecated]
            str(stat_file), stat_name_col=5
        )
    assert list(df.index) == ["ADCS", "BBBB"]
    assert list(df["lat"]) == [-43.2, -42.2]


def test_load_station_file(tmp_path: Path) -> None:
    station_file = tmp_path / "stations.ll"
    station_file.write_text("171.1 -43.2 ADCS\n172.1\t-42.2   BBBB\n")

    with pytest.deprecated_call():
        df = formats.load_station_file(str(station_file))  # ty: ignore[deprecated]

    assert list(df.columns) == ["lon", "lat"]
    assert list(df.index) == ["ADCS", "BBBB"]
    assert df.loc["ADCS", "lon"] == pytest.approx(171.1)
    assert df.loc["ADCS", "lat"] == pytest.approx(-43.2)
    assert df.loc["BBBB", "lon"] == pytest.approx(172.1)
    assert df.loc["BBBB", "lat"] == pytest.approx(-42.2)


def test_load_station_ll_vs30(tmp_path: Path) -> None:
    station_file = tmp_path / "stations.ll"
    station_file.write_text("171.1 -43.2 ADCS\n172.1 -42.2 BBBB\n")
    vs30_file = tmp_path / "stations.vs30"
    vs30_file.write_text("ADCS 250.0\nBBBB 500.0\n")

    with pytest.deprecated_call():
        df = formats.load_station_ll_vs30(  # ty: ignore[deprecated]
            str(station_file), str(vs30_file)
        )

    expected = pd.DataFrame(
        {
            "vs30": [250.0, 500.0],
            "lon": [171.1, 172.1],
            "lat": [-43.2, -42.2],
        },
        index=pd.Index(["ADCS", "BBBB"]),
    )
    pd.testing.assert_frame_equal(
        df.sort_index(), expected, check_names=False, check_index_type=False
    )
