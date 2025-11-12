"""Utility functions common to many CLI scripts."""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import (
    Annotated,
    ParamSpec,
    TypeVar,  # <- Import TypeVar
    get_args,
    get_origin,
)

import docstring_parser
import typer
from docstring_parser.common import DocstringStyle
from typer.models import ArgumentInfo, OptionInfo

# P captures the parameters (args and kwargs) of the decorated function.
P = ParamSpec("P")

# R captures the return type of the decorated function.
R = TypeVar("R")

T = ParamSpec("T")


def from_docstring(
    app: typer.Typer,
    **kwargs: T.kwargs,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Apply help texts from the function's docstring to Typer arguments/options and command.

    Parameters
    ----------
    app : typer.Typer
        The Typer application to which the command will be registered.
    **kwargs : Any
        Additional keyword arguments to be passed to the Typer command.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        A decorator function that takes a command (Callable[P, R]) and
        returns a wrapper (Callable[P, R]) preserving its signature P and return R.
    """

    def decorator(command: Callable[P, R]) -> Callable[P, R]:  # numpydoc ignore=GL08
        if command.__doc__ is None:
            return command

        # Parse the docstring and extract parameter descriptions
        docstring = docstring_parser.parse(
            command.__doc__, style=DocstringStyle.NUMPYDOC
        )
        param_help = {param.arg_name: param.description for param in docstring.params}

        # The command's full help text (summary + long description)
        command_help = (
            f"{docstring.short_description or ''}\n\n{docstring.long_description or ''}"
        )

        # Get the signature of the original function
        sig = inspect.signature(command)
        parameters = sig.parameters

        # Prepare a new mapping for parameters
        new_parameters = []

        for name, param in parameters.items():
            help_text = param_help.get(
                name, ""
            )  # Get help text from docstring if available
            param_type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else str
            )

            # Handle Annotated types
            if get_origin(param_type) is Annotated:
                param_type, *metadata = get_args(param_type)
                new_metadata = []
                for m in metadata:
                    if isinstance(m, ArgumentInfo | OptionInfo):
                        if not m.help:
                            m.help = help_text
                    new_metadata.append(m)
                new_param = param.replace(
                    annotation=Annotated[param_type, *new_metadata]
                )

            # If it's an Option or Argument directly
            elif isinstance(param.default, ArgumentInfo | OptionInfo):
                if not param.default.help:
                    param.default.help = help_text
                new_param = param

            else:
                # If the parameter has no default, treat it as an Argument
                if param.default is inspect.Parameter.empty:
                    new_param = param.replace(
                        default=typer.Argument(..., help=help_text),
                        annotation=param_type,
                    )
                else:
                    # If the parameter has a default, treat it as an Option
                    new_param = param.replace(
                        default=typer.Option(param.default, help=help_text),
                        annotation=param_type,
                    )

            new_parameters.append(new_param)

        # Create a new signature with updated parameters
        new_sig = sig.replace(parameters=new_parameters)

        # Register the command with the app
        # Since the signature (P, R) is applied to the decorator result,
        # the wrapper's type definition must match what command returns (R).
        @app.command(help=command_help.strip(), **kwargs)
        @wraps(command)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # numpydoc ignore=GL08
            return command(*args, **kwargs)

        # NOTE: Typer requires the dynamic signature update for runtime reflection,
        # but the type checker uses the P, R generics.
        wrapper.__signature__ = new_sig

        return wrapper

    return decorator
