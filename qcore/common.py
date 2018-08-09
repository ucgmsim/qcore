"""
Contains generic functionality common to im-processing
"""


def is_virtual_station(station_name):
    """
    station_name: (string / unicode)
    Checks if all restraints on virtual station names are met:
    1) Virtual Stations have 7 characters
    2) Virtual Stations contain no capitals
    3) Virtual Stations must be valid hex strings
    """
    # 7 characters long
    if len(station_name) != 7:
        return False

    n_caps = sum(1 for c in station_name if c.isupper())
    if n_caps > 0:
        return False

    # valid hex string
    try:
        if not isinstance(station_name, int):
            int(station_name, 16)
    except (ValueError, TypeError):
        return False

    # all tests passed
    return True
