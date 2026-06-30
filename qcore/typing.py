"""Typing utilities."""

from typing import Any, TypeVar

import numpy as np

TNFloat = TypeVar("TNFloat", bound=np.floating[Any])
TFloat = TypeVar("TFloat", bound=np.floating[Any] | float)
