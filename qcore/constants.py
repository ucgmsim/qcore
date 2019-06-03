from enum import Enum
from datetime import datetime
from typing import List

LF_DEFAULT_NCORES = 160  # 4 nodes, no hyperthreading
CHECKPOINT_DURATION = 10.0 # in minutes

HF_DEFAULT_NCORES = 80  # 1 node, hyperthreading
HF_DEFAULT_VERSION = "run_hf_mpi"

BB_DEFAULT_VERSION = "run_bb_mpi"
BB_DEFAULT_NCORES = 80  # 1 node, hyperthreading

IM_CALC_DEFAULT_N_CORES = 40  # 1 node, no hyperthreading
IM_CALC_COMPONENTS = ["geom", "000", "090", "ver", "ellipsis"]

IM_SIM_CALC_TEMPLATE_NAME = "sim_im_calc.sl.template"
IM_SIM_SL_SCRIPT_NAME = "sim_im_calc_{}.sl"

MERGE_TS_DEFAULT_NCORES = 4

HEADER_TEMPLATE = "slurm_header.cfg"
DEFAULT_ACCOUNT = "nesi00213"
DEFAULT_MEMORY = "16G"

# Why do we have to different time formats?
METADATA_TIMESTAMP_FMT = "%Y-%m-%d_%H:%M:%S"
METADATA_LOG_FILENAME = "metadata_log.json"

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

EST_MODEL_NN_PREFIX = "model_NN_"
EST_MODEL_SVR_PREFIX = "model_SVR_"

SLURM_MGMT_DB_NAME = "slurm_mgmt.db"

VM_PARAMS_FILE_NAME = "vm_params.yaml"

ROOT_DEFAULTS_FILE_NAME = "root_defaults.yaml"


class EstModelType(Enum):
    NN = "NN"
    SVR = "SVR"
    NN_SVR = "NN_SVR"


class HPC(Enum):
    maui = "maui"
    mahuika = "mahuika"


class ExtendedEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class ExtendedStrEnum(ExtendedEnum):
    @classmethod
    def has_str_value(cls, str_value):
        return any(str_value == item.str_value for item in cls)

    @classmethod
    def from_str(cls, str_value):
        if not cls.has_str_value(str_value):
            raise ValueError(
                "{} is not a valid {}".format(str_value, ProcessType.__name__)
            )
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
    """

    EMOD3D = (
        1,
        "EMOD3D",
        False,
        False,
        'srun {emod3d_bin} -args "par={lf_sim_dir}/e3d.par"',
        (),
    )
    merge_ts = (
        2,
        "merge_ts",
        True,
        False,
        "time srun {merge_ts_path} filelist=$filelist outfile=$OUTFILE nfiles=$NFILES",
        (1,),
    )
    HF = (
        4,
        "HF",
        True,
        True,
        "srun python $gmsim/workflow/scripts/hf_sim.py {fd_statlist} {hf_bin_path} -m {v_mod_1d_name} --duration "
        "{duration} --dt {dt} --sim_bin {sim_bin_path}",
        (),
    )
    BB = (
        5,
        "BB",
        True,
        True,
        "srun python $gmsim/workflow/scripts/bb_sim.py {outbin_dir} {vel_mod_dir} {hf_bin_path} {stat_vs_est} "
        "{bb_bin_path} --flo {flo}",
        (1, 4),
    )
    IM_calculation = (
        6,
        "IM_calc",
        False,
        False,
        "time python $IMPATH/calculate_ims.py {sim_dir}/BB/Acc/BB.bin b -o {sim_dir}/IM_calc/ -np {np} -i "
        "{sim_name} -r {fault_name} -c {component} -t s {extended} {simple}",
        ((5,), (12,), (13,)),
    )
    IM_plot = 7, None, None, False, None, (6,)
    rrup = 8, "rrup", None, False, None, ()
    Empirical = 9, None, None, False, None, (8,)
    Verification = 10, None, None, False, None, (9,)
    clean_up = 11, "clean_up", None, None, None, (6, )
    LF2BB = 12, "LF2BB", None, None, None, (1,)
    HF2BB = 13, "HF2BB", None, None, None, (4,)

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

    @classmethod
    def get_by_name(cls, name):
        for _, member in cls.__members__.items():
            if member.str_value == name:
                return member
        raise LookupError

    def get_remaining_dependencies(self, completed_dependencies: List['ProcessType'] = ()) -> List[int]:
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
                message.append("The tasks {} are mutually exclusive and cannot be run at the same time.\n".format(
                    (x.str_value for x in task_group)
                ))
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

    im_pSA_count = "pSA_count"
    im_comp = "im_components"
    im_comp_count = "im_components_count"


class Components(ExtendedEnum):
    geom = "geom"
    c000 = "000"
    c090 = "090"
    ver = "ver"
    ellipsis = "ellipsis"


class Status(ExtendedStrEnum):
    """Job status on the HPC"""

    created = 1, "created"
    queued = 2, "queued"
    running = 3, "running"
    completed = 4, "completed"
    failed = 5, "failed"
    unknown = 6, "unknown"

    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj


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

    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj


class ERFFileType(ExtendedStrEnum):
    nhm = 0, "nhm"

    def __new__(cls, value, str_value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj


