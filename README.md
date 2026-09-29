[![Build Status](https://quakecoresoft.canterbury.ac.nz/jenkins/job/qcore/badge/icon?build=last:${params.ghprbActualCommit=master)](https://quakecoresoft.canterbury.ac.nz/jenkins/job/qcore)

## Installation:

To install run `pip install -e .` from the root directory of this repository

The `qcore.cli` helpers (`from_docstring`) need the optional `cli` extra, which pulls in typer and docstring_parser: `pip install 'qcore-utils[cli]'` (or `pip install -e '.[cli]'`).

### Downloading data after install

After installing, if you intend to use GMT for plotting you will need to run "download_data.py" (Located in qcore.data) to download and
unpack the data used for these scripts
