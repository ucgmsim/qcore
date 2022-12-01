from enum import Enum, auto
from functools import wraps
from os.path import abspath

import numpy as np

DISK_DTYPE = "<f4"
SINGLE_DTYPE = np.single


def _check_data_exists(method):
    """
    A wrapper to ensure that a velocity model array exists before you do anything to it
    This can either be from creating a new one, or opening an existing one
    The wraps decorator ensures the properties of the wrapped function are passed to help etc correctly
    """

    @wraps(method)
    def inner(vm_obj, *args, **kwargs):
        assert vm_obj._data is not None, "Must open file or create new model first"
        return method(vm_obj, *args, **kwargs)

    return inner


def _check_data_empty(method):
    """
    A wrapper to ensure that a velocity model array does not exist before you do anything to it
    The wraps decorator ensures the properties of the wrapped function are passed to help etc correctly
    """

    @wraps(method)
    def inner(vm_obj, *args, **kwargs):
        assert vm_obj._data is None, (
            "Must not have a loaded model. Call close() to unload the model. "
            "Do not forget to save the current model first if you wish to keep it"
        )
        return method(vm_obj, *args, **kwargs)

    return inner


def create_constant_vm_file(pert_f_location, npoints, value=1):
    """
    Creates and saves a vm file with given size and uniform value
    :param pert_f_location: The location to save the file to
    :param npoints: The size of the file to make
    :param value: The value to set the model to. Defaults to 1
    """
    # Doesn't matter what the dimensions are, we only need the total point count
    with VelocityModelFile(npoints, 1, 1) as vmf:
        vmf.set_values(np.full(vmf.shape, value))
        vmf.save(pert_f_location)


class DataState(Enum):
    """
    Represents the ways of storing velocity model data
    NUMPY: nx, ny, nz
    EMOD3D: ny, nz, nx
    """

    NUMPY = auto()
    EMOD3D = auto()


class VelocityModelFile:
    """
    A representation of an EMOD3D style velocity model file (VMF).
    Supports use as a context manager and allowing easy creation and editing of velocity model files.
    Internal memory type and disk storage type are single precision float
    Some large velocity models may be larger than the available ram. memmap should be implemented if this becomes an issue.
    """

    _data: np.ndarray = None

    def __init__(
        self, nx: int, ny: int, nz: int, file_loc=None, writable=False, memmap=False
    ):
        """
        Creates a object that represents a velocity model binary file as used by EMOD3D
        These files contain nx*ny*nz single precision floats, stored in the order ny, nz, nx
        :param nx: The nx dimension of the velocity model file
        :param ny: The ny dimension of the velocity model file
        :param nz: The nz dimension of the velocity model file
        :param file_loc: The path to the file. This can be either an input for an existing file or the destination of a file being created
        :param writable: Allow the given file path to be written to
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.read_only = not writable

        if file_loc is not None:
            self.file_path = abspath(file_loc)
        else:
            self.file_path = None

        self._data_state = None

        self._memmap = memmap

    def __enter__(self):
        """
        If a filepath was given opens the filepath
        If a filepath was not given creates a new empty model
        Returns the VMF object
        """
        if self._data is None:
            if self.file_path is None:
                self.new()
            else:
                self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @_check_data_empty
    def open(self, memmap=None):
        """
        Opens the given velocity model file and loads it into memory
        Converts from EMOD3D format to NUMPY format
        This may need to change to memmap if large file loading becomes a problem
        """
        if self.file_path is None:
            raise ValueError(
                "Must set file path. Or use new() to create a new model from scratch"
            )
        if memmap is not None:
            self._memmap = memmap

        if self._memmap:
            mode = "r" if self.read_only else "r+"
            self._data = np.memmap(
                self.file_path, dtype=DISK_DTYPE, mode=mode, shape=self.emod_shape
            )
        else:
            self._data = np.fromfile(self.file_path, DISK_DTYPE).reshape(
                self.emod_shape
            )
        self._data_state = DataState.EMOD3D
        self._change_data_state(DataState.NUMPY)

    @_check_data_empty
    def new(self, memmap=None, filepath=None):
        """
        Creates a new velocity model of 0s.
        Must not have a model loaded.
        """

        if memmap is not None:
            self._memmap = memmap

        if self._memmap:
            if filepath is not None:
                self.file_path = filepath
            if self.file_path is None:
                raise AttributeError("filepath must be set for memmap mode to be used")
            self._data = np.memmap(
                filepath, shape=self.shape, dtype=DISK_DTYPE, mode="w+"
            )
            self._data.fill(0)
        else:
            self._data = np.zeros(self.shape, dtype=SINGLE_DTYPE)
        self._data_state = DataState.NUMPY
        self.read_only = False

    @_check_data_exists
    def _change_data_state(self, target_state: DataState):
        """
        Changes the state of the data internally, handling all current state transitions
        Doesn't do anything if the current and target state are the same
        :param target_state: The target state. Must be a DataState enum member
        """
        if not isinstance(target_state, DataState):
            raise TypeError("Must use DataState type to change data state")

        # Ensure we have the right state transition. Future proof in case of more states
        if self._data_state == DataState.NUMPY and target_state == DataState.EMOD3D:
            self._data = np.swapaxes(self._data, 1, 2)
            self._data = np.swapaxes(self._data, 0, 2)
            self._data_state = DataState.EMOD3D

        elif self._data_state == DataState.EMOD3D and target_state == DataState.NUMPY:
            self._data = np.swapaxes(self._data, 0, 2)
            self._data = np.swapaxes(self._data, 1, 2)
            self._data_state = DataState.NUMPY

    @_check_data_exists
    def save(self, fp=None):
        """
        Saves the matrix to the given file, or overwrites the input file by default
        Saves in the standard EMOD3D format
        :param fp: The filepath to save the file to. By default overwrites the
        """
        if self.file_path is None and fp is None:
            raise AttributeError("A filepath must be given to save the data to")
        elif fp is None:
            fp = self.file_path

        if abspath(fp) == self.file_path and self.read_only:
            raise AttributeError(
                f"File {self.file_path} is set to read only. "
                f"Save the file to another location or reload the file with read only disabled"
            )

        self._change_data_state(DataState.EMOD3D)
        self._data.tofile(fp)
        self._change_data_state(DataState.NUMPY)

    @_check_data_exists
    def close(self):
        """Closes the file without saving"""
        self._data = None

    @_check_data_exists
    def get_value(self, x, y, z):
        return self._data[x, y, z]

    @_check_data_exists
    def set_value(self, value, x, y, z):
        """
        Sets the value of the given location to the given value
        Caution: This operation does not write to disk, a manual save() call must be performed
        :param value: The new value to be set. Will be cast to a single precision float
        :param x, y, z: The coordinate in the x, y, z directions respectively. May use negative indicies
        """
        self._data[x, y, z] = value

    @_check_data_exists
    def multiply_values(self, value):
        """
        Multiplies all the values in the velocity model by the values in the given array or velocity model
        This can be used to manually generate perturbated velocity models
        :param value: A scalar, numpy broadcastable array, array with the same shape or VelocityModelFile to multiply by
        """
        if isinstance(value, VelocityModelFile):
            with value:
                value = value._data
        self._data = self._data * value

    @property
    def shape(self):
        """
        Returns the shape of the underlying data
        :return: A numpy shape style tuple
        """
        return self.nx, self.ny, self.nz

    @property
    def emod_shape(self):
        """
        Returns the shape of the underlying array in emod3d order
        :return: A numpy shape style tuple
        """
        return self.ny, self.nz, self.nx

    @_check_data_empty
    def set_values(self, values: np.ndarray, state=DataState.NUMPY):
        """
        Sets the underlying data to the given array, ensuring it has the correct size
        Must not have a model loaded, as doing so would discard the
        Caution: Does not write the values to disk, save() must be called manually
        :param values: Array with values to be stored. Cast to single precision floating point
        :param state: The order the values are stored in. Normally this is standard (nx, ny, nz)
        """
        assert (
            values.shape == self.shape
        ), f"Shapes don't match: {values.shape}, {self.shape}"
        self._data = values
        self._data_state = state

    @_check_data_exists
    def get_values(self):
        return self._data

    @_check_data_exists
    def apply_limits(self, lower: float = -np.inf, upper: float = np.inf):
        """
        Applys limits to the values in the velocity model. By default all values are permitted
        :param lower:
        :param upper:
        """
        self._data = np.minimum(upper, np.maximum(lower, self._data))
