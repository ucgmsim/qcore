from enum import Enum, auto
from datetime import datetime
from typing import List

import numpy as np

CHECKPOINT_DURATION = 10.0

QUEUE_DATE_FORMAT = "%Y%m%d%H%M%S_%f"

# Why do we have to different time formats?
METADATA_TIMESTAMP_FMT = "%Y-%m-%d_%H:%M:%S"
METADATA_LOG_FILENAME = "metadata_log.json"

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

EST_MODEL_NN_PREFIX = "model_NN_"
EST_MODEL_SVR_PREFIX = "model_SVR_"

SLURM_MGMT_DB_NAME = "slurm_mgmt.db"

VM_PARAMS_FILE_NAME = "vm_params.yaml"

IM_SIM_CALC_INFO_SUFFIX = "_imcalc.info"

ROOT_DEFAULTS_FILE_NAME = "root_defaults.yaml"

HF_DEFAULT_SEED = 0

MAXIMUM_EMOD3D_TIMESHIFT_1_VERSION = "3.0.4"

DEFAULT_PSA_PERIODS = [
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
]
EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.0), num=100, base=10)


class EstModelType(Enum):
    NN = "NN"
    SVR = "SVR"
    NN_SVR = "NN_SVR"


class ExtendedEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

    @classmethod
    def is_substring(cls, parent_string):
        """Check if an enum's string value is contained in the given string"""
        return any(item.value in parent_string for item in cls)

    @classmethod
    def get_names(cls):
        return [item.name for item in cls]

    def __str__(self):
        return self.name


class ExtendedStrEnum(ExtendedEnum):
    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj

    @classmethod
    def has_str_value(cls, str_value):
        return any(str_value == item.str_value for item in cls)

    @classmethod
    def from_str(cls, str_value):
        if not cls.has_str_value(str_value):
            raise ValueError("{} is not a valid {}".format(str_value, cls.__name__))
        else:
            for item in cls:
                if item.str_value == str_value:
                    return item

    @classmethod
    def iterate_str_values(cls, ignore_none=True):
        """Iterates over the string values of the enum,
        ignores entries without a string value by default
        """
        for item in cls:
            if ignore_none and item.str_value is None:
                continue
            yield item.str_value


# Process 1-5 are simulation 6-7 are Intensity Measure and 8-10 are simulation verification
class ProcessType(ExtendedStrEnum):
    """Constants for process type. Int values are used in python workflow,
    str values are for metadata collection and estimator training (str values used
    in the estimator configs)

    The string value of the enum can be accessed with Process.EMOD3D.str_value

    # ProcessID, ProcessName, "Does this task use Hyperthreading?", "Does this use an Acc directory?", command, dependancies (tuple),
    """

    EMOD3D = (
        1,
        "EMOD3D",
        False,
        False,
        '{run_command} {emod3d_bin} -args "par={lf_sim_dir}/e3d.par"',
        (),
    )
    merge_ts = (
        2,
        "merge_ts",
        True,
        False,
        "time {run_command} {merge_ts_path} filelist=$filelist outfile=$OUTFILE nfiles=$NFILES",
        (1,),
    )

    plot_ts = (3, "plot_ts", True, None, None, (2,))

    HF = (
        4,
        "HF",
        True,
        True,
        "{run_command} python $gmsim/workflow/scripts/hf_sim.py {fd_statlist} {hf_bin_path} -m {hf_vel_mod_1d} --duration "
        "{duration} --dt {dt} --sim_bin {sim_bin_path}",
        (),
    )
    BB = (
        5,
        "BB",
        True,
        True,
        "{run_command} python $gmsim/workflow/scripts/bb_sim.py {outbin_dir} {vel_mod_dir} {hf_bin_path} {stat_vs_est} "
        "{bb_bin_path} --flo {flo}",
        (1, 4),
    )
    IM_calculation = (
        6,
        "IM_calc",
        False,
        False,
        "time python $IMPATH/calculate_ims.py {sim_dir}/BB/Acc/BB.bin b -o {sim_dir}/IM_calc/ -np {np} -i "
        "{sim_name} -r {fault_name} -t s {component} {extended} {simple} {advanced_IM} {pSA_periods}",
        ((5,), (12,), (13,)),
    )
    IM_plot = 7, "IM_plot", None, False, None, (6,)
    rrup = 8, "rrup", None, False, None, ()
    Empirical = 9, "Empirical", None, False, None, (8,)
    Verification = 10, None, None, False, None, (9,)
    clean_up = 11, "clean_up", None, None, None, (6,)
    LF2BB = 12, "LF2BB", None, None, None, (1,)
    HF2BB = 13, "HF2BB", None, None, None, (4,)
    plot_srf = 14, "plot_srf", None, False, None, ()

    # adv_im uses the same base code as IM_calc
    advanced_IM = (15, "advanced_IM") + IM_calculation[2:]

    def __new__(
        cls, value, str_value, is_hyperth, uses_acc, command_template, dependencies
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        obj.is_hyperth = is_hyperth
        obj.uses_acc = uses_acc
        obj.command_template = command_template
        obj.dependencies = dependencies
        return obj

    def get_remaining_dependencies(
        self, completed_dependencies: List["ProcessType"] = ()
    ) -> List[int]:
        """Determines if the task has any unmet dependencies and returns a list of them if so. Only does single level
        dependencies, does not recurse
        :param completed_dependencies: Tasks that have been completed and therefore may contribute to this tasks
        dependencies
        :return: A list of integers representing the unmet dependency tasks.
        """
        dependencies = self.dependencies
        if len(self.dependencies) > 0 and not isinstance(self.dependencies[0], int):
            if any(
                (
                    all(
                        (
                            ProcessType(dependency) in completed_dependencies
                            for dependency in multi_dependency
                        )
                    )
                    for multi_dependency in self.dependencies
                )
            ):
                # At least one of the dependency conditions for the task is fulfilled, no need to add any more tasks
                return []
            # Otherwise the first dependency list is the default
            dependencies = self.dependencies[0]
        return [x for x in dependencies if ProcessType(x) not in completed_dependencies]

    @staticmethod
    def check_mutually_exclusive_tasks(tasks):
        """If multiple tasks from any of the given groups are specified, then the simulation cannot run
        :param tasks: The list of tasks to be run
        :return: A string containing any errors found during the check"""
        mutually_exclusive_tasks = (
            (ProcessType.BB, ProcessType.LF2BB, ProcessType.HF2BB),
        )
        message = []
        for task_group in mutually_exclusive_tasks:
            if len([x for x in task_group if x in tasks]) > 1:
                message.append(
                    "The tasks {} are mutually exclusive and cannot be run at the same time.\n".format(
                        (x.str_value for x in task_group)
                    )
                )
        return "\n".join(message)


class MetadataField(ExtendedEnum):
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
    n_steps = "n_steps"
    start_time = "start_time"
    end_time = "end_time"
    submit_time = "submit_time"
    status = "status"

    im_pSA_count = "pSA_count"
    im_comp = "im_components"
    im_comp_count = "im_components_count"


class Status(ExtendedStrEnum):
    """Job status on the HPC"""

    created = 1, "created"
    queued = 2, "queued"
    running = 3, "running"
    unknown = 4, "unknown"
    completed = 5, "completed"
    failed = 6, "failed"


class RootParams(Enum):
    """Keywords for the root yaml file.
    Note: These are not complete!
    """

    flo = "flo"
    dt = "dt"
    version = "version"
    stat_file = "stat_file"
    stat_vs_est = "stat_vs_est"
    stat_vs_ref = "stat_vs_ref"
    v_mod_1d_name = "v_mod_1d_name"
    mgmt_db_location = "mgmt_db_location"
    seed = "seed"
    extended_period = "extended_period"


class FaultParams(Enum):
    """Keywords for the fault yaml file.
    Note: These are not complete!
    """

    root_yaml_path = "root_yaml_path"
    vel_mod_dir = "vel_mod_dir"
    stat_coords = "stat_coords"
    FD_STATLIST = "FD_STATLIST"


class SimParams(Enum):
    """Keywords for the simulation yaml file.
    Note: These are not complete!
    """

    fault_yaml_path = "fault_yaml_path"
    run_name = "run_name"
    user_root = "user_root"
    run_dir = "run_dir"
    sim_dir = "sim_dir"
    srf_file = "srf_file"
    vm_params = "vm_params"
    sim_duration = "sim_duration"
    slip = "slip"
    stat_file = "stat_file"


class VMParams(Enum):
    """Keywords for the vm params yaml file."""

    model_lat = "MODEL_LAT"
    model_lon = "MODEL_LON"
    model_rot = "MODEL_ROT"
    hh = "hh"
    nx = "nx"
    ny = "ny"
    nz = "nz"
    sufx = "sufx"
    gridfile = "GRIDFILE"
    gridout = "GRIDOUT"
    model_coords = "MODEL_COORDS"
    model_params = "MODEL_PARAMS"
    model_bounds = "MODEL_BOUNDS"
    extent_x = "extent_x"
    extent_y = "extent_y"
    extent_zmax = "extent_zmax"
    extent_zmin = "extent_zmin"


class HazardType(Enum):
    """The different hazard fault types"""

    emp_a = "emp_a"
    emp_b = "emp_b"
    emp_ds = "emp_ds"
    emp_tot = "emp_tot"
    cs_a = "cs_a"
    cs_tot = "cs_tot"


class SourceToSiteDist(ExtendedStrEnum):
    R_rup = 0, "r_rup"
    R_jb = 1, "r_jb"
    R_x = 2, "r_x"
    R_y = 3, "r_y"
    Backarc = 4, "backarc"


class Components(ExtendedStrEnum):
    c090 = 0, "090"
    c000 = 1, "000"
    cver = 2, "ver"
    cgeom = 3, "geom"
    crotd50 = 4, "rotd50"
    crotd100 = 5, "rotd100"
    crotd100_50 = 6, "rotd100_50"
    cnorm = 7, "norm"

    @staticmethod
    def get_comps_to_calc_and_store(arg_comps: List[str]):
        """
        convert arg comps to str_comps for integer_conversion in read_waveform & str comps for writing result
        :param arg_comps: user input a list of comp(s)
        :return: two lists of str comps
        """

        def component_sorter(x):
            return x.value

        components_to_store = [Components.from_str(c) for c in arg_comps]
        components_to_store.sort(key=component_sorter)

        horizontal_components = set(list(Components)[:2])
        basic_components = set(list(Components)[:3])
        advanced_components = set(list(Components)[3:])
        advanced_components_to_get = list(
            advanced_components.intersection(set(components_to_store))
        )

        if advanced_components_to_get:
            components_to_get = list(
                basic_components.intersection(
                    set(components_to_store) | horizontal_components
                )
            )
            components_to_get.sort(key=component_sorter)
        else:
            components_to_get = components_to_store[:]

        return components_to_get, components_to_store


class PLATFORM_CONFIG(Enum):
    LF_DEFAULT_NCORES = auto()
    HF_DEFAULT_NCORES = auto()
    HF_DEFAULT_VERSION = auto()
    BB_DEFAULT_VERSION = auto()
    BB_DEFAULT_NCORES = auto()
    IM_CALC_DEFAULT_N_CORES = auto()
    IM_SIM_CALC_TEMPLATE_NAME = auto()
    IM_SIM_SL_SCRIPT_NAME = auto()
    MERGE_TS_DEFAULT_NCORES = auto()
    DEFAULT_ACCOUNT = auto()
    DEFAULT_MEMORY = auto()
    MACHINE_TASKS = auto()
    DEFAULT_N_RUNS = auto()
    SCHEDULER = auto()
    AVAILABLE_MACHINES = auto()
    ESTIMATION_MODELS_DIR = auto()
    TEMPLATES_DIR = auto()
    VELOCITY_MODEL_DIR = auto()
    RUN_COMMAND = auto()
    HEADER_FILE = auto()
