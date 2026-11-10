from pathlib import Path

import pandas as pd
import pytest

from qcore import formats


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
