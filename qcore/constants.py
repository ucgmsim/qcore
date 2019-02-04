from enum import Enum


# Process 1-5 are simulation 6-7 are Intensity Measure and 8-10 are simulation verification
class ProcessType(Enum):
    """Constants for process type. Int values are used in python workflow,
    str values are for metadata collection and estimator training (str values used
    in the estimator configs)

    The string value of the enum can be accessed with Process.EMOD3D.str_value
    """
    EMOD3D = 1, "EMOD3D"
    merge_ts = 2, "merge_ts"
    winbin_aio = 3, None
    HF = 4, "HF"
    BB = 5, "BB"
    IM_calculation = 6, "IM_calc"
    IM_plot = 7, None
    rrup = 8, None
    Empirical = 9, None
    Verification = 10, None

    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj


    @classmethod
    def has_str_value(cls, value):
        return any(value == item.str_value for item in cls)

    @classmethod
    def iterate_str_values(cls, ignore_none: bool = True):
        """Iterates over the string values of the enum,
        ignores entries without a string value by default
        """
        for item in cls:
            if ignore_none and item.str_value is None:
                continue
            yield item.str_value


