from os.path import exists

import numpy as np

SINGLE_DTYPE = np.single


def _check_data(method):
    """
    A wrapper to ensure that a velocity model array exists before you do anything to it
    This can either be from creating a new one, or opening an existing one
    """

    def inner(vm_obj, *args):
        assert vm_obj._data is not None, "Must open file first"
        return method(vm_obj, *args)

    return inner


class VelocityModelFile:

    _data = None

    def __init__(self, nx, ny, nz, file_loc=None):
        """
        Creates a object that represents a velocity model binary file as used by EMOD3D
        These files contain nx*ny*nz single precision floats, stored in the order ny, nz, nx
        :param nx: The nx dimension of the velocity model file
        :param ny: The ny dimension of the velocity model file
        :param nz: The nz dimension of the velocity model file
        :param file_loc: The path to the file. This can be either an input for an existing file or the destination of a file being created
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.file_path = file_loc

    def __enter__(self):
        if self._data is None:
            self.open()
        return self._data

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self, fp=None):
        if fp is not None:
            self.file_path = fp

        if self.file_path is None:
            raise ValueError(
                "Must set file path. Or use new() to create a new file from scratch"
            )

        if exists(self.file_path):
            self._data = np.fromfile(self.file_path, "<f4").reshape(
                [self.ny, self.nz, self.nx]
            )
            self._data = np.swapaxes(self._data, 0, 2)
            self._data = np.swapaxes(self._data, 1, 2)
        else:
            self.new()

    def new(self):
        self._data = np.zeros((self.nx, self.ny, self.nz), dtype=SINGLE_DTYPE)

    @_check_data
    def save(self, fp=None):
        """Saves the matrix to the given file, or over writes the input file by default"""
        if fp is None:
            fp = self.file_path
        self._data = np.swapaxes(self._data, 1, 2)
        self._data = np.swapaxes(self._data, 0, 2)
        self._data.tofile(fp)
        self._data = np.swapaxes(self._data, 0, 2)
        self._data = np.swapaxes(self._data, 1, 2)

    @_check_data
    def close(self):
        """Saves and closes the file"""
        self._data = None

    @_check_data
    def get_value(self, x, y, z):
        if self._data is None:
            raise ValueError("Must open file first")
        return self._data[x, y, z]

    @_check_data
    def set_value(self, value, x, y, z):
        self._data[x, y, z] = value

    @_check_data
    def multiply_values(self, value):
        self._data = self._data * value

    @_check_data
    def set_values(self, values: np.ndarray):
        assert (
            values.shape == self._data.shape
        ), f"Shapes don't match: {values.shape}, {self._data.shape}"
        self._data = values.astype(SINGLE_DTYPE)
