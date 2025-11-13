import os
import tempfile
from pathlib import Path

import pytest
import yaml

from qcore import utils


def test_load_yaml():
    # Create a temporary YAML file
    data = {"key1": "value1", "key2": 123, "key3": ["a", "b", "c"]}
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        yaml.safe_dump(data, f)
        yaml_file = f.name
    
    try:
        result = utils.load_yaml(yaml_file)
        assert result == data
        assert result["key1"] == "value1"
        assert result["key2"] == 123
        assert result["key3"] == ["a", "b", "c"]
    finally:
        Path(yaml_file).unlink()


def test_dump_yaml():
    data = {"test": "data", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
        output_file = f.name
    
    try:
        utils.dump_yaml(data, output_file)
        
        # Verify the file was created and contains correct data
        with open(output_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == data
    finally:
        Path(output_file).unlink()


def test_setup_dir_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_subdir")
        assert not os.path.exists(test_dir)
        
        utils.setup_dir(test_dir)
        
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)


def test_setup_dir_empty_removes_contents():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_subdir")
        os.makedirs(test_dir)
        
        # Create a file in the directory
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        assert os.path.exists(test_file)
        
        # Call with empty=True should remove and recreate
        utils.setup_dir(test_dir, empty=True)
        
        assert os.path.exists(test_dir)
        assert not os.path.exists(test_file)


def test_setup_dir_existing_without_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "test_subdir")
        os.makedirs(test_dir)
        
        # Create a file
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Call without empty=True should preserve contents
        utils.setup_dir(test_dir, empty=False)
        
        assert os.path.exists(test_dir)
        assert os.path.exists(test_file)


@pytest.mark.parametrize(
    "version1,version2,expected",
    [
        ("1.0.0", "1.0.0", 0),
        ("1.0.1", "1.0.0", 1),
        ("1.0.0", "1.0.1", -1),
        ("1.1.0", "1.0.9", 1),
        ("2.0.0", "1.9.9", 1),
        ("1.0.1", "1.0", 1),
        ("1.0", "1.0.1", -1),
        ("1.1", "1", 1),
        ("1", "1.1", -1),
    ],
)
def test_compare_versions(version1: str, version2: str, expected: int):
    result = utils.compare_versions(version1, version2)
    assert result == expected


def test_compare_versions_with_letters():
    # Test that non-numeric characters are stripped
    result = utils.compare_versions("v1.0.0", "1.0.0")
    assert result == 0
    
    result = utils.compare_versions("v1.1.0", "v1.0.0")
    assert result == 1


def test_compare_versions_custom_separator():
    result = utils.compare_versions("1-0-1", "1-0-0", split_char="-")
    assert result == 1
