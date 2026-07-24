"""
Gives access to the folder structure of the cybershake directory
"""

import os


def get_fault_from_realisation(realisation: str) -> str:
    """
    Extract the fault name from a realisation name or path.

    Parameters
    ----------
    realisation : str
        Realisation name or full path to the realisation.

    Returns
    -------
    str
        Fault name associated with the given realisation.
    """
    realisation = os.path.basename(realisation)  # if realisation is a fullpath
    return realisation.rsplit("_REL", 1)[0]


def get_realisation_name(fault_name: str, rel_no: int) -> str:
    """
    Format a realisation name given a fault name and realisation number.

    Parameters
    ----------
    fault_name : str
        Name of the fault.
    rel_no : int
        Realisation number.

    Returns
    -------
    str
        Formatted realisation name (e.g., 'AlpineF2_REL03').
    """
    return f"{fault_name}_REL{rel_no:0>2}"


def get_srf_info_location(realisation: str) -> str:
    """
    Get the relative SRF info file location for a realisation.

    Parameters
    ----------
    realisation : str
        Realisation name.

    Returns
    -------
    str
        Relative path to the SRF info file.
    """
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".info")


def get_srf_dir(cybershake_root: str, realisation: str) -> str:
    """
    Get the directory containing SRF files for a fault.

    Parameters
    ----------
    cybershake_root : str
        Cybershake root directory.
    realisation : str
        Realisation name.

    Returns
    -------
    str
        Path to the SRF directory for the fault.
    """
    return os.path.join(
        cybershake_root,
        "Data",
        "Sources",
        get_fault_from_realisation(realisation),
        "Srf",
    )


def get_srf_location(realisation: str) -> str:
    """
    Get the relative SRF file location for a realisation.

    Parameters
    ----------
    realisation : str
        Realisation name.

    Returns
    -------
    str
        Relative path to the SRF file.
    """
    fault = get_fault_from_realisation(realisation)
    return os.path.join(fault, "Srf", realisation + ".srf")


def get_srf_path(cybershake_root: str, realisation: str) -> str:
    """
    Get the absolute path to the SRF file for a realisation.

    Parameters
    ----------
    cybershake_root : str
        Cybershake root directory.
    realisation : str
        Realisation name.

    Returns
    -------
    str
        Path to the SRF file.
    """
    return os.path.join(
        cybershake_root, "Data", "Sources", get_srf_location(realisation)
    )


def get_fault_dir(cybershake_root: str, fault_name: str) -> str:
    """
    Get the directory for a specific fault's simulations.

    Parameters
    ----------
    cybershake_root : str
        Cybershake root directory.
    fault_name : str
        Fault name.

    Returns
    -------
    str
        Path to the fault's directory within 'Runs'.
    """
    return os.path.join(cybershake_root, "Runs", fault_name)


def get_sim_dir(cybershake_root: str, realisation: str) -> str:
    """
    Get the simulation directory for a specific realisation.

    Parameters
    ----------
    cybershake_root : str
        Cybershake root directory.
    realisation : str
        Realisation name.

    Returns
    -------
    str
        Path to the simulation directory.
    """
    return os.path.join(
        get_fault_dir(cybershake_root, get_fault_from_realisation(realisation)),
        realisation,
    )


def get_im_calc_dir(sim_root: str, realisation: str | None = None) -> str:
    """Get IM Calc directory recursively.

    Parameters
    ----------
    sim_root : str
        Path to the simulation root directory.
    realisation : str, optional
        Realisation to fetch IM calc directory for.

    Returns
    -------
    str
        Path to the IM calc directory.
    """
    if realisation is None:
        return os.path.join(sim_root, "IM_calc")
    else:
        return get_im_calc_dir(get_sim_dir(sim_root, realisation))


def get_IM_csv_from_root(cybershake_root: str, realisation: str) -> str:  # noqa: N802
    """Get IM csv file for realisation.

    Parameters
    ----------
    cybershake_root : str
        Cybershake root directory.
    realisation : str
        Realisation name.

    Returns
    -------
    str
        Path to the realisation's IM csv file.
    """
    return os.path.join(
        get_im_calc_dir(get_sim_dir(cybershake_root, realisation)),
        "{}.{}".format(realisation, "csv"),
    )


def get_fault_yaml_path(sim_root: str, fault_name: str | None = None) -> str:
    """
    Gets the fault_params.yaml for the specified simulation.

    For the manual workflow set fault_name to None as the fault params
    are stored directly in the simulation directory.

    Parameters
    ----------
    sim_root : str
        The simulation root directory.
    fault_name : str, optional
        The fault name, or None.

    Returns
    -------
    str
        The path to the fault_params.yaml for the given fault.
    """
    return os.path.join(sim_root, fault_name or "", "fault_params.yaml")
