import re
from pathlib import PurePath, Path
from typing import Optional


import pooch
import filelock
import requests


def get_latest_registry_version() -> str:
    """Get the latest registry version from the main branch of the registry repository.

    Returns
    -------
    str
        The latest sha256 hash of the main branch in the ucgmsim/registry repo.
    """
    with requests.get(
        "https://api.github.com/repos/ucgmsim/registry/commits/main", timeout=30
    ) as repo_data:
        return repo_data.json()[
            "sha"
        ]  # Get the latest commit hash from the main branch of the registry


def qcore_registry(
    cache_path: Path = pooch.os_cache("quakecore"),
    registry: Optional[dict[str, str]] = None,
    reference: Optional[str] = None,
) -> pooch.Pooch:
    """
    Creates and returns a Pooch registry object for managing file downloads.

    Parameters
    ----------
    cache_path : Path, optional
        The directory where files will be cached. Defaults to the OS cache
        directory for "quakecore".
    registry : dict[str, str], optional
        Registry database to use. A registry database consists of paths
        relative to the root of the registry repo as keys, and sha256
        checksums as values.
    reference : str, optional
        The commit hash, branch or tag (in git parlance, a reference) to
        source the data files for. If not supplied, will use latest commit
        on the `main` branch.

    Returns
    -------
    pooch.Pooch
        A Pooch object configured to manage the quakecore registry.
    """
    if not reference:
        reference = get_latest_registry_version()

    with requests.get(
        f"https://raw.githubusercontent.com/ucgmsim/registry/{reference}/registry.txt",
        timeout=30,
    ) as registry_file:
        registry = dict(
            tuple(reversed(tuple(re.split(" +", line.strip()))))
            for line in registry_file.text.splitlines()
            if line.strip()
        )

    qcore_pooch = pooch.create(
        path=cache_path,
        base_url=f"https://raw.githubusercontent.com/ucgmsim/registry/{reference}",
        version=f"0.0+{reference}",
        registry=registry,
    )

    return qcore_pooch


def fetch_file(
    registry: pooch.Pooch, filepath: PurePath, lock_timeout: int = 60
) -> Path:
    """Return a path to the pooch registry file.

    This function checks if pooch would download or update the requested
    file path. If it does, then it acquire a lock on this file and pooch
    performs the action. Otherwise we simply return the file.

    Parameters
    ----------
    registry : pooch.Pooch
        The registry to download the file from.
    filepath : PurePath
        The registry file path to download (relative to the root of the registry).
    lock_timeout : int, default = 60
        The file lock timeout.

    Returns
    -------
    Path
        The path to the downloaded file in local storage.
    """
    local_path = registry.abspath / filepath
    known_hash = registry.registry[local_path]
    if pooch.core.download_action(local_path, known_hash) != "fetch":
        lock_path = registry.abspath / filepath.with_suffix(filepath.suffix + ".lock")
        with filelock.FileLock(lock_path, timeout=lock_timeout):
            return Path(registry.fetch(filepath))
    else:
        return Path(registry.fetch(filepath))
