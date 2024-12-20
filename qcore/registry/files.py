from importlib import resources
from pathlib import Path
from typing import Optional

import pooch


def build_registry(
    cache_path: Path = pooch.os_cache("quakecore"),
    registry: Optional[dict[str, str]] = None,
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
    qcore_pooch = pooch.create(
        path=cache_path,
        base_url="http://hypocentre:9999",
        registry=registry,
    )

    if not registry:
        with (
            resources.files("qcore.registry")
            .joinpath("registry.txt")
            .open("r") as registry_file
        ):
            qcore_pooch.load_registry(registry_file)
    return qcore_pooch


def get_file(file_name: str, version: str, cache_path: Optional[Path] = None) -> Path:
    """
    Retrieves a file from the registry based on its name and version.

    Parameters
    ----------
    file_name : str
        The name of the file to retrieve.
    version : str
        The version of the file to retrieve.
    cache_path : Path, optional
        The directory where files will be cached. If not provided, the default
        cache path for "quakecore" is used.

    Returns
    -------
    Path
        The local file path to the downloaded file.
    """
    if cache_path:
        registry = build_registry(cache_path)
    else:
        registry = build_registry()
    return Path(registry.fetch(f"{version}/{file_name}"))
