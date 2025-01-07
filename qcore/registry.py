import re
from pathlib import Path
from typing import Optional

import pooch
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
    version: Optional[str] = None,
) -> pooch.Pooch:
    """
    Creates and returns a Pooch registry object for managing file downloads.

    Parameters
    ----------
    cache_path : Path, optional
        The directory where files will be cached. Defaults to the OS cache
        directory for "quakecore".
    registry : dict[str, str], optional
        Registry database to use. Consists of
    Returns
    -------
    pooch.Pooch
        A Pooch object configured to manage the quakecore registry.
    """
    if not version:
        version = get_latest_registry_version()

    with requests.get(
        f"https://raw.githubusercontent.com/ucgmsim/registry/{version}/registry.txt",
        timeout=30,
    ) as registry_file:
        registry = dict(
            tuple(reversed(tuple(re.split(" +", line.strip()))))
            for line in registry_file.text.splitlines()
            if line.strip()
        )

    qcore_pooch = pooch.create(
        path=cache_path,
        base_url=f"https://raw.githubusercontent.com/ucgmsim/registry/{version}",
        version=f"0.0+{version}",
        registry=registry,
    )

    return qcore_pooch
