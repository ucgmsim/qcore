"""DEPRECATED - Global constants and Enum helper."""

from collections.abc import Generator
from enum import Enum
from typing import Any
from warnings import deprecated  # type: ignore


@deprecated("Use built-in Enum")
class ExtendedEnum(Enum):
    """DEPRECATED: Utility enum extension. Use built-in Enum."""

    @classmethod
    def has_value(cls, value: Any) -> bool:
        """Check if enum has value.

        Parameters
        ----------
        value : Any
            Value to check.

        Returns
        -------
        bool
            True if Enum has value.
        """
        return any(value == item.value for item in cls)

    @classmethod
    def is_substring(cls, parent_string: str) -> bool:
        """Check if an enum's string value is contained in the given string

        Parameters
        ----------
        parent_string : str
            The string to search for the value.

        Returns
        -------
        bool
            True if any enum value is a substring of `parent_string`.
        """
        return any(
            not isinstance(item.value, str) or item.value in parent_string
            for item in cls
        )

    @classmethod
    def get_names(cls) -> list[str]:
        """Get the names for every enum member.

        Returns
        -------
        list of str
            The enum member names.
        """
        return [item.name for item in cls]

    def __str__(self) -> str:
        """Get a string representation of Enum value.

        Returns
        -------
        str
            The enum member name.
        """
        return self.name


@deprecated("Use built-in StrEnum")
class ExtendedStrEnum(ExtendedEnum):  # type: ignore
    """DEPRECATED: Utility Enum extension for string mappings. Use built-in StrEnum."""

    def __new__(cls, value: Any, str_value: str):  # noqa: D102 # numpydoc ignore=GL08
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj

    @classmethod
    def has_str_value(cls, str_value: str) -> bool:
        """Check if any enum member has a given string value.

        Parameters
        ----------
        str_value : str
            The value to check.

        Returns
        -------
        bool
            True if the Enum contains a member whose string value matches `str_value`.
        """
        return any(str_value == item.str_value for item in cls)

    @classmethod
    def from_str(cls, str_value: str) -> Any:
        """Lookup an enum member from its str_value.

        Parameters
        ----------
        str_value : str
            The string value of the enum member.

        Returns
        -------
        Any
            The enum member with matching `str_value`.

        Raises
        ------
        ValueError
            If the enum does not contain a member with the given string value.
        """
        for item in cls:
            if item.str_value == str_value:
                return item
        raise ValueError(f"{str_value} is not a valid {cls.__name__}")

    @classmethod
    def iterate_str_values(cls, ignore_none: bool = True) -> Generator[Any, None, None]:
        """Iterates over the member variables of the enum.

        Parameters
        ----------
        ignore_none : bool
            If True, ignore all member variables whose str_value is None.

        Yields
        ------
        Any
            An enum member variable.
        """
        yield from (
            item for item in cls if not (ignore_none and item.str_value is None)
        )
