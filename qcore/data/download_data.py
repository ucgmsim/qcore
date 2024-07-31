import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Annotated
from urllib import request

import typer

app = typer.Typer()

DATA_VERSION = "1.2"
DATA_URL = "https://www.dropbox.com/scl/fi/p5so9irbx43c9g96jr2qr/qcore_resources.tar.xz?rlkey=pgmn69e212c0zx0fauxuikgky&dl=1"


def extract_data(archive, destination):
    with tarfile.open(archive) as xz:
        xz.extractall(destination)


def get_version(version_path):
    if os.path.isfile(version_path):
        return open(version_path).read().strip()
    else:
        return None


@app.command(help="Downloads the qcore data and extracts it")
def download_data(
    download_location: Annotated[
        Path, typer.Option(help="Location of the qcore/data directory")
    ] = Path(__file__).parent,
):
    loc_version = download_location / "version"
    # Get the current version
    have_ver = get_version(loc_version)

    # Check if the data is already downloaded or a different version
    if str(have_ver) != DATA_VERSION:
        print("data package missing or incorrect version")
        print("downloading...")
        with tempfile.NamedTemporaryFile() as data_archive_file:
            request.urlretrieve(DATA_URL, data_archive_file.name)
            extract_data(data_archive_file.name, download_location.parent)

    # Version Check
    have_ver = get_version(loc_version)
    if str(have_ver) != DATA_VERSION:
        sys.exit("data package issue, please contact repository maintainer")


if __name__ == "__main__":
    app()
