import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qcore import registry


def test_resolve_git_reference_with_commit_hash():
    # Test that a full commit hash is returned as-is
    commit_hash = "a" * 40  # 40 character hash
    result = registry.resolve_git_reference(commit_hash)
    assert result == commit_hash


def test_resolve_git_reference_with_longer_hash():
    # Test that a longer commit hash (e.g., 64 chars) is returned as-is
    commit_hash = "a" * 64
    result = registry.resolve_git_reference(commit_hash)
    assert result == commit_hash


@patch("qcore.registry.requests.get")
def test_resolve_git_reference_with_branch(mock_get):
    # Test that a branch name is resolved to a commit hash
    mock_response = MagicMock()
    mock_response.json.return_value = {"sha": "b" * 40}
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_get.return_value = mock_response
    
    result = registry.resolve_git_reference("main")
    assert result == "b" * 40
    mock_get.assert_called_once()


@patch("qcore.registry.requests.get")
def test_qcore_registry_default_behavior(mock_get):
    # Test creating a qcore registry with default parameters
    mock_response = MagicMock()
    mock_response.json.return_value = {"sha": "c" * 40}
    mock_response.text = "file1.txt  abc123def456\nfile2.txt  789ghi012jkl\n"
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_get.return_value = mock_response
    
    result = registry.qcore_registry()
    
    assert result is not None
    # Verify it's a Pooch object
    assert hasattr(result, "fetch")
    assert hasattr(result, "registry")


@patch("qcore.registry.requests.get")
def test_qcore_registry_with_custom_reference(mock_get):
    # Test creating registry with a specific reference
    commit_hash = "d" * 40
    mock_response = MagicMock()
    mock_response.text = "file1.txt  abc123\n"
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_get.return_value = mock_response
    
    result = registry.qcore_registry(reference=commit_hash)
    
    assert result is not None
    # The reference should be used in the base_url
    assert commit_hash in result.base_url


@patch("qcore.registry.requests.get")
def test_qcore_registry_with_custom_registry(mock_get):
    # Test creating registry with a custom registry dict
    custom_registry = {
        "test_file.txt": "abc123def456",
        "another_file.txt": "789ghi012jkl",
    }
    commit_hash = "e" * 40
    
    result = registry.qcore_registry(reference=commit_hash, registry=custom_registry)
    
    assert result is not None
    assert "test_file.txt" in result.registry
    assert result.registry["test_file.txt"] == "abc123def456"


def test_fetch_file_basic():
    # Test basic file fetching (mock the pooch behavior)
    mock_pooch = MagicMock()
    mock_pooch.abspath = Path("/tmp/cache")
    mock_pooch.registry = {"test.txt": "abc123"}
    mock_pooch.fetch.return_value = "/tmp/cache/test.txt"
    
    with patch("qcore.registry.core.download_action", return_value=("update", None)):
        with patch("qcore.registry.filelock.FileLock"):
            result = registry.fetch_file(mock_pooch, Path("test.txt"))
    
    assert result == Path("/tmp/cache/test.txt")
    mock_pooch.fetch.assert_called_once_with("test.txt")


def test_fetch_file_no_download_needed():
    # Test when file doesn't need downloading
    mock_pooch = MagicMock()
    mock_pooch.abspath = Path("/tmp/cache")
    mock_pooch.registry = {"test.txt": "abc123"}
    mock_pooch.fetch.return_value = "/tmp/cache/test.txt"
    
    with patch("qcore.registry.core.download_action", return_value=("fetch", None)):
        result = registry.fetch_file(mock_pooch, Path("test.txt"))
    
    assert result == Path("/tmp/cache/test.txt")
    mock_pooch.fetch.assert_called_once()
