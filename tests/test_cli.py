from typing import Annotated, Optional

import pytest
import typer
from typer.testing import CliRunner

from qcore import cli

runner = CliRunner()

DOCSTRING = """Example command.

Parameters
----------
param1 : int
    This is the first parameter.
param2 : Optional[str]
    This is an optional parameter.
"""


def test_from_docstring(capsys: pytest.CaptureFixture[str]):
    """Test the from_docstring decorator applies help texts correctly."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def example_command(
        param1: Annotated[int, typer.Argument()],
        param2: Annotated[Optional[str], typer.Option()] = "a",
    ) -> None:
        """Example command.

        Parameters
        ----------
        param1 : int
            This is the first parameter.
        param2 : Optional[str]
            This is an optional parameter.
        """
        print("Hello World", param1, param2)

    # Ensure the docstring is unchanged
    assert example_command.__doc__ == DOCSTRING
    # Run `--help` and check output

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Example command." in result.output
    assert "This is the first parameter." in result.output
    assert "This is an optional parameter." in result.output
    result = runner.invoke(app, ["0", "--param2", "b"])
    assert result.stdout.strip() == "Hello World 0 b"

    example_command(1, "c")
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello World 1 c"


def test_from_docstring_kwargs(capsys: pytest.CaptureFixture[str]):
    """Test the from_docstring decorator passes kwargs correctly."""

    app = typer.Typer()

    @cli.from_docstring(app, name="command_1")
    def example_command(
        param1: Annotated[int, typer.Argument()],
        param2: Annotated[Optional[str], typer.Option()] = "a",
    ) -> None:
        """Example command.

        Parameters
        ----------
        param1 : int
            This is the first parameter.
        param2 : Optional[str]
            This is an optional parameter.
        """
        print("Hello World", param1, param2)

    @cli.from_docstring(app, name="command_2")
    def example_command_2(
        param1: Annotated[int, typer.Argument()],
        param2: Annotated[Optional[str], typer.Option()] = "b",
    ) -> None:
        """Example command 2.

        Parameters
        ----------
        param1 : int
            This is the first parameter of command 2.
        param2 : Optional[str]
            This is an optional parameter for command 2.
        """
        print("Hello World from command 2", param1, param2)

    # Ensure the docstring is unchanged
    assert example_command.__doc__ == DOCSTRING
    # Run `--help` and check output

    result = runner.invoke(app, ["command_1", "--help"])
    assert result.exit_code == 0
    assert "Example command." in result.output
    assert "This is the first parameter." in result.output
    assert "This is an optional parameter." in result.output

    result = runner.invoke(app, ["command_2", "--help"])
    assert result.exit_code == 0
    assert "Example command 2." in result.output
    assert "This is the first parameter of command 2." in result.output
    assert "This is an optional parameter for command 2." in result.output

    result = runner.invoke(app, ["command_1", "0", "--param2", "b"])
    assert result.stdout.strip() == "Hello World 0 b"

    result = runner.invoke(app, ["command_2", "0", "--param2", "b"])
    assert result.stdout.strip() == "Hello World from command 2 0 b"

    example_command(1, "c")
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello World 1 c"
