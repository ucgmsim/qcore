from enum import Enum
from datetime import datetime

LF_DEFAULT_NCORES = 160  # 4 nodes, no hyperthreading


HF_DEFAULT_NCORES = 80  # 1 node, hyperthreading
HF_DEFAULT_VERSION = "run_hf_mpi"

BB_DEFAULT_VERSION = "run_bb_mpi"
BB_DEFAULT_NCORES = 80  # 1 node, hyperthreading

IM_CALC_DEFAULT_N_PROCESSES = 40
IM_CALC_COMPONENTS = ["geom", "000", "090", "ver", "ellipsis"]

IM_SIM_CALC_TEMPLATE_NAME = "sim_im_calc.sl.template"
IM_SIM_SL_SCRIPT_NAME = "sim_im_calc_{}.sl"

MERGE_TS_DEFAULT_NCORES = 4

HEADER_TEMPLATE = "slurm_header.cfg"
DEFAULT_ACCOUNT = "nesi00213"
DEFAULT_MEMORY = "16G"

# Why do we have to different time formats?
METADATA_TIMESTAMP_FMT = "%Y-%m-%d_%H:%M:%S"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


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


class MetadataField(Enum):
    sim_name = "sim_name"
    run_time = "run_time"
    core_hours = "core_hours"
    n_cores = "cores"
    fd_count = "fd_count"
    nsub_stoch = "nsub_stoch"
    dt = "dt"
    nt = "nt"
    nx = "nx"
    ny = "ny"
    nz = "nz"
    start_time = "start_time"
    end_time = "end_time"

    im_pSA_count = "pSA_count"
    im_comp = "im_components"
    im_comp_count = "im_components_count"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)
