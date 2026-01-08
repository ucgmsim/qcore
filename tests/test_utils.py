import tempfile
from pathlib import Path

import yaml

from qcore import utils


def test_load_yaml() -> None:
    # Create a temporary YAML file
    data = {"key1": "value1", "key2": 123, "key3": ["a", "b", "c"]}

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        yaml.safe_dump(data, f)
        yaml_file = f.name

    try:
        result = utils.load_yaml(yaml_file)  # type: ignore
        assert result == data
        assert result["key1"] == "value1"
        assert result["key2"] == 123
        assert result["key3"] == ["a", "b", "c"]
    finally:
        Path(yaml_file).unlink()
