import os
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib import request

PACKAGE_NAME = "qcore"
PACKAGE_URL = f"https://github.com/ucgmsim/{PACKAGE_NAME}"
DATA_VERSION = "1.2"
DATA_NAME = "qcore_resources.tar.xz"
DATA_URL = f"{PACKAGE_URL}/releases/download/{DATA_VERSION}/{DATA_NAME}"


def extract_data(archive, destination):
    with tarfile.open(archive) as xz:
        xz.extractall(destination)


def get_version(version_path: Path):
    if not version_path.exists():
        return None
    with open(version_path, encoding="utf-8") as infile:
        return infile.read().strip()


def download_data():
    download_location = Path(__file__).parent
    loc_version = download_location / "version"

    have_ver = get_version(loc_version)
    if str(have_ver) != DATA_VERSION:
        print("data package missing or incorrect version")
        print("downloading...")
        with tempfile.NamedTemporaryFile() as data_archive_file:
            request.urlretrieve(DATA_URL, data_archive_file.name)
            extract_data(data_archive_file.name, download_location.parent)
    # final check
    have_ver = get_version(loc_version)
    if str(have_ver) != DATA_VERSION:
        sys.exit("data package issue, please contact repository maintainer")


if __name__ == "__main__":
    download_data()
