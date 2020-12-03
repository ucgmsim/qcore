import sys
import os
from shutil import rmtree

from qcore import shared

"""
Functions used during testing by other repositories.
Expected to be called in the style of

@pytest.yield_fixture(scope="session", autouse=True)
def set_up(request):
    data_locations = utils.test_set_up(REALISATIONS)
    yield list(zip(data_locations, [rel[0] for rel in REALISATIONS]))
    utils.test_tear_down(data_locations) 
"""


def test_set_up(realizations):
    """
    Downloads test data and places each set in a separate folder, then returns the list of test data locations
    :param realizations: A list of (realisation name, data zip url) pairs to download data for testing
    :return: A list of test data locations, containing the contents of the downloaded zip files
    """
    test_data_save_dirs = []
    for i, (realization, data_download_path) in enumerate(realizations):
        data_store_path = os.path.join(os.getcwd(), "sample" + str(i))
        zip_download_path = os.path.join(data_store_path, realization + ".zip")

        download_cmd = "wget -O {} {}".format(zip_download_path, data_download_path)
        unzip_cmd = "unzip {} -d {}".format(zip_download_path, data_store_path)
        # print(DATA_STORE_PATH)
        test_data_save_dirs.append(data_store_path)
        if not os.path.isdir(data_store_path):
            os.makedirs(data_store_path, exist_ok=True)
            out, err = shared.exe(download_cmd, debug=False)
            if "error" in err:
                rmtree(data_store_path)
                sys.exit("{} failed to retrieve test data".format(err))
            # download_via_ftp(DATA_DOWNLOAD_PATH, zip_download_path)
            if not os.path.isfile(zip_download_path):
                sys.exit(
                    "File failed to download from {}. Exiting".format(
                        data_download_path
                    )
                )
            out, err = shared.exe(unzip_cmd, debug=False)
            os.remove(zip_download_path)
            if "error" in err:
                rmtree(data_store_path)
                sys.exit("{} failed to extract data folder".format(err))

        else:
            print("Benchmark data folder already exits: ", data_store_path)
    print(test_data_save_dirs)
    # Run all tests
    return test_data_save_dirs


def test_tear_down(test_data_save_dirs):
    """
    Removes all the given test data locations
    :param test_data_save_dirs: a list of test data locations. Same as return value of test_set_up
    """
    for test_data_dir in test_data_save_dirs:
        if os.path.isdir(test_data_dir):
            rmtree(test_data_dir, ignore_errors=True)
