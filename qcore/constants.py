from enum import Enum, auto
from datetime import datetime
from typing import List, Tuple, Union

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


class DependencyTarget(Enum):
    MEDIAN = auto()
    REL = auto()


class Dependency:
    """
    A ProcessType, DependencyTarget pair.
    With just in time generation to prevent circular declarations with the ProcessType enum
    """

    _process: "ProcessType"
    _dependency: DependencyTarget

    def __init__(
        self,
        proc_id: Union[int, tuple, "ProcessType"],
        dependency_target: DependencyTarget = DependencyTarget.REL,
    ):
        self._process = proc_id
        self._dependency = dependency_target

    @property
    def process(self):
        if isinstance(self._process, int):
            self._process = ProcessType(self._process)
        elif isinstance(self._process, tuple):
            self._process = ProcessType(self._process[0])
        return self._process

    @property
    def dependency(self):
        return self._dependency

    def __str__(self):
        return f"{self.process.name}_{self.dependency.name}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            isinstance(other, Dependency)
            and self.process == other.process
            and self.dependency == other.dependency
        )


# Process 1-5 are simulation 6-7 are Intensity Measure and 8-10 are simulation verification
class ProcessType(ExtendedStrEnum):
    """Constants for process type. Int values are used in python workflow,
    str values are for metadata collection and estimator training (str values used
    in the estimator configs)

    The string value of the enum can be accessed with Process.EMOD3D.str_value

    # ProcessID, ProcessName, "Does this task use Hyperthreading?", "Does this use an Acc directory?", command, dependancies (tuple),
    """

    VM_PARAMS = 16, "VM_PARAMS", None, False, None, ([],)
    VM_GEN = (
        17,
        "VM_GEN",
        None,
        False,
        None,
        ([(VM_PARAMS, "MEDIAN")],),
    )
    VM_PERT = (
        18,
        "VM_PERT",
        None,
        False,
        None,
        ([(VM_PARAMS, "MEDIAN")],),
    )  # Needs VM_params generated for REL_1
    INSTALL_FAULT = (
        19,
        "INSTALL_FAULT",
        None,
        False,
        None,
        ([(VM_GEN, "MEDIAN")],),
    )

    EMOD3D = (
        1,
        "EMOD3D",
        False,
        False,
        '{run_command} {emod3d_bin} -args "par={lf_sim_dir}/e3d.par"',
        ([], [(INSTALL_FAULT, "MEDIAN")]),
    )
    merge_ts = (
        2,
        "merge_ts",
        True,
        False,
        "time {run_command} {merge_ts_path} filelist=$filelist outfile=$OUTFILE nfiles=$NFILES",
        ([(EMOD3D, "REL")],),
    )

    plot_ts = (
        3,
        "plot_ts",
        True,
        None,
        None,
        ([(merge_ts, "REL")],),
    )

    HF = (
        4,
        "HF",
        True,
        True,
        "{run_command} python $gmsim/workflow/workflow/calculation/hf_sim.py {fd_statlist} {hf_bin_path} --duration "
        "{duration} --dt {dt} --sim_bin {sim_bin_path}",
        ([], [(INSTALL_FAULT, "MEDIAN")]),
    )
    BB = (
        5,
        "BB",
        True,
        True,
        "{run_command} python $gmsim/workflow/workflow/calculation/bb_sim.py {outbin_dir} {vel_mod_dir} {hf_bin_path} {stat_vs_est} "
        "{bb_bin_path} --flo {flo}",
        (
            [
                (EMOD3D, "REL"),
                (HF, "REL"),
            ],
        ),
    )
    LF2BB = (
        12,
        "LF2BB",
        None,
        None,
        None,
        ([(EMOD3D, "REL")],),
    )
    HF2BB = 13, "HF2BB", None, None, None, ([(HF, "REL")],)
    IM_calculation = (
        6,
        "IM_calc",
        False,
        False,
        "time python $IMPATH/calculate_ims.py {sim_dir}/BB/Acc/BB.bin b -o {sim_dir}/IM_calc/ -np {np} -i "
        "{sim_name} -r {fault_name} -t s {component} {extended} {simple} {advanced_IM} {pSA_periods}",
        (
            [(BB, "REL")],
            [(LF2BB, "REL")],
            [(HF2BB, "REL")],
        ),
    )
    advanced_IM = (15, "advanced_IM") + IM_calculation[2:]
    IM_plot = (
        7,
        "IM_plot",
        None,
        False,
        None,
        ([(IM_calculation, "REL")],),
    )
    rrup = (
        8,
        "rrup",
        None,
        False,
        None,
        ([], [(INSTALL_FAULT, "REL")]),
    )
    Empirical = (
        9,
        "Empirical",
        None,
        False,
        None,
        ([(rrup, "REL")],),
    )
    Verification = (
        10,
        None,
        None,
        False,
        None,
        ([(Empirical, "REL")],),
    )
    clean_up = (
        11,
        "clean_up",
        None,
        None,
        None,
        ([(IM_calculation, "REL")],),
    )

    plot_srf = (
        14,
        "plot_srf",
        None,
        False,
        None,
        ([], [(INSTALL_FAULT, "REL")]),
    )
    # adv_im uses the same base code as IM_calc

    def __new__(
        cls, value, str_value, is_hyperth, uses_acc, command_template, dependencies
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        obj.is_hyperth = is_hyperth
        obj.uses_acc = uses_acc
        obj.command_template = command_template
        obj.dependencies = [
            [Dependency(proc, DependencyTarget[dep]) for proc, dep in dependency_set]
            for dependency_set in dependencies
        ]
        return obj

    def get_remaining_dependencies(
        self, completed_dependencies: List[Dependency] = ()
    ) -> List[Dependency]:
        """Determines if the task has any unmet dependencies and returns a list of them if so. Only does single level
        dependencies, does not recurse
        :param completed_dependencies: Tasks that have been completed and therefore may contribute to this tasks
        dependencies
        :return: A list of integers representing the unmet dependency tasks.
        """
        if any(
            (
                all((x in completed_dependencies for x in multi_dependency))
                for multi_dependency in self.dependencies
            )
        ):
            # At least one of the dependency conditions for the task is fulfilled, no need to add any more tasks
            return []
        # Otherwise the first dependency list is the default
        return [x for x in self.dependencies[0] if x not in completed_dependencies]

    @staticmethod
    def check_mutually_exclusive_tasks(tasks: List["ProcessType"]):
        """If multiple tasks from any of the given groups are specified, then the simulation cannot run
        TODO: Add in more complex handling so that only mutually exclusive tasks with overlapping SQL matches get caught
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
    killed_WCT = 6, "killed_WCT"
    failed = 7, "failed"


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
    component = "component"


class FaultParams(Enum):
    """Keywords for the fault yaml file.
    Note: These are not complete!
    """

    root_yaml_path = "root_yaml_path"
    vel_mod_dir = "vel_mod_dir"
    stat_coords = "stat_coords"
    FD_STATLIST = "FD_STATLIST"
    site_v1d_dir = "site_v1d_dir"
    site_specific = "site_specific"


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


class SourceToSiteDist(ExtendedStrEnum):
    R_rup = 0, "r_rup"
    R_jb = 1, "r_jb"
    R_x = 2, "r_x"
    R_y = 3, "r_y"
    Backarc = 4, "backarc"


class Components(ExtendedStrEnum):
    """
    Intensity measure components.
    Basic intensity measures (090, 000, ver, H1, H2) are those available in ascii format from observed records.
    Advanced/derived intensity measures (geom, norm, rotX) are those derived from basic intensity measures.
    """

    c090 = 0, "090"
    c000 = 1, "000"
    cver = 2, "ver"
    ch1 = 3, "H1"
    ch2 = 4, "H2"

    cgeom = 5, "geom"
    crotd50 = 6, "rotd50"
    crotd100 = 7, "rotd100"
    crotd100_50 = 8, "rotd100_50"
    cnorm = 9, "norm"
    ceas = 10, "EAS"

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

        # The horizontal components generated by simulation workflow steps
        horizontal_components = {Components.c090, Components.c000}
        basic_components = set(Components.get_basic_components())

        advanced_components = set(list(Components)[5:])

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

    @staticmethod
    def get_basic_components():
        """
        Components with ascii observed records
        :return: The list of basic components
        """
        return [
            Components.c090,
            Components.c000,
            Components.cver,
            Components.ch1,
            Components.ch2,
        ]


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
    VM_PERT_DEFAULT_NCORES = auto()
    DEFAULT_ACCOUNT = auto()
    PLATFORM_ACCOUNTS = auto()
    DEFAULT_MEMORY = auto()
    MACHINE_TASKS = auto()
    DEFAULT_N_RUNS = auto()
    SCHEDULER = auto()
    AVAILABLE_MACHINES = auto()
    ESTIMATION_MODELS_DIR = auto()
    GMSIM_TEMPLATES_DIR = auto()
    SCHEDULER_TEMPLATES_DIR = auto()
    EXAMPLES_DIR = auto()
    VELOCITY_MODEL_DIR = auto()
    RUN_COMMAND = auto()
    HEADER_FILE = auto()
    DEFAULT_SITE_RESPONSE_DIR = auto()


class SlHdrOptConsts(Enum):
    job_name_prefix = "job_name_prefix"
    account = "account"
    n_tasks = "n_tasks"
    memory = "memory"
    additional = "additional_lines"
    wallclock = "wallclock_limit"
    version = "version"
    description = "job_description"


class SlBodyOptConsts(Enum):
    component = "component"
    n_procs = "np"
    sim_dir = "sim_dir"
    sim_name = "sim_name"
    fault_name = "fault_name"
    output_dir = "output_dir"
    extended = "extended"
    simple_out = "simple"
    advanced_IM = ProcessType.advanced_IM.str_value
    mgmt_db = "mgmt_db"


class VelocityModelTopographyType(ExtendedEnum):
    true = "TRUE"
    bulldozed = "BULLDOZED"
    squashed = "SQUASHED"
    squashed_tapered = "SQUASHED_TAPERED"
