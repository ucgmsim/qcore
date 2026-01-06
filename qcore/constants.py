"""DEPRECATED - Global constants and Enum helper."""

from collections.abc import Generator
from enum import Enum
from typing import Any
from warnings import deprecated  # type: ignore


@deprecated
class ExtendedEnum(Enum):
    """DEPRECATED: Utility enum extension. Use built-in Enum."""

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


@deprecated
class ExtendedStrEnum(ExtendedEnum):
    """DEPRECATED: Utility Enum extension for string mappings. Use built-in StrEnum."""

    def __new__(cls, value: Any, str_value: str):  # noqa: D102 # numpydoc ignore=GL08
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


        """
        )
