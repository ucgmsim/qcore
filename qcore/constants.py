"""DEPRECATED - Global constants and Enum helper."""

from collections.abc import Generator
from enum import Enum
from typing import Any

from typing_extensions import deprecated  # type: ignore


@deprecated("Use built-in Enum")
class ExtendedEnum(Enum):
    """DEPRECATED: Utility enum extension. Use built-in Enum."""

    @classmethod
    def has_value(cls, value: Any) -> bool:
        return any(value == item.value for item in cls)

    @classmethod
    def is_substring(cls, parent_string: str) -> bool:
        """Check if an enum's string value is contained in the given string"""
        return any(
            not isinstance(item.value, str) or item.value in parent_string
            for item in cls
        )

    @classmethod
    def get_names(cls) -> list[str]:
        return [item.name for item in cls]

    def __str__(self) -> str:
        return self.name


@deprecated("Use built-in StrEnum")
class ExtendedStrEnum(ExtendedEnum):  # type: ignore
    """DEPRECATED: Utility Enum extension for string mappings. Use built-in StrEnum."""

    _value_: Any
    str_value: str

    def __new__(cls, value: Any, str_value: str):  # noqa: D102 # numpydoc ignore=GL08
        obj = object.__new__(cls)
        obj._value_ = value
        obj.str_value = str_value
        return obj

    @classmethod
    def has_str_value(cls, str_value: str) -> bool:
        return any(str_value == item.str_value for item in cls)

    @classmethod
    def from_str(cls, str_value):
        if not cls.has_str_value(str_value):
            raise ValueError(f"{str_value} is not a valid {cls.__name__}")
        else:
            for item in cls:
                if item.str_value == str_value:
                    return item

    @classmethod
    def iterate_str_values(cls, ignore_none: bool = True) -> Generator[Any, None, None]:
        """Iterates over the string values of the enum,
        ignores entries without a string value by default
        """
        for item in cls:
            if ignore_none and item.str_value is None:
                continue
            yield item.str_value
