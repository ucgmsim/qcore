import os
from setuptools import setup
import sys
import tarfile
from urllib.request import urlretrieve


PACKAGE_NAME = "qcore"
PACKAGE_URL = f"https://github.com/ucgmsim/{PACKAGE_NAME}"
DATA_VERSION = "1.2"
DATA_NAME = "qcore_resources.tar.xz"
DATA_URL = f"{PACKAGE_URL}/releases/download/{DATA_VERSION}/{DATA_NAME}"

NO_DATA_ARG = "--no-data"


def extract_data(archive, destination):
    with tarfile.open(archive) as xz:
        xz.extractall(destination)


def get_version(version_path):
    if os.path.isfile(version_path):
        return open(version_path).read().strip()
    else:
        return None


def prepare_data():
    loc_version = os.path.join(PACKAGE_NAME, "data", "version")
    # extract existing archive
    have_ver = get_version(loc_version)
    if str(have_ver) != DATA_VERSION and os.path.isfile(DATA_NAME):
        # extract available archive
        print("checking available archive version...")
        extract_data(DATA_NAME, PACKAGE_NAME)
    # download missing archive
    have_ver = get_version(loc_version)
    print(have_ver)
    if str(have_ver) != DATA_VERSION:
        print("data package missing or incorrect version")
        print("downloading...")
        urlretrieve(DATA_URL, DATA_NAME)
        extract_data(DATA_NAME, PACKAGE_NAME)
    # final check
    have_ver = get_version(loc_version)
    if str(have_ver) != DATA_VERSION:
        sys.exit("data package issue, please contact repository maintainer")


# TODO: Temporary fix to keep Jenkins testing going
if NO_DATA_ARG in sys.argv:
    print(f"Skip downloading {DATA_NAME}")
    sys.argv.remove(NO_DATA_ARG)
else:
    prepare_data()

setup(
    name=PACKAGE_NAME,
    version="1.2",
    packages=[PACKAGE_NAME],
    url=PACKAGE_URL,
    description="QuakeCoRE Library",
    package_data={
        "qcore": [os.path.join("configs", "*.json"), "data/*", "data/*/*", "data/*/*/*"]
    },
    include_package_data=True,
    install_requires=["numpy<2.0", "scipy>=0.16", "dataclasses"],
)
