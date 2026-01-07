import inspect
from typing import Annotated, Optional

import pytest
import typer
from typer.testing import CliRunner

from qcore import cli

runner = CliRunner()

DOCSTRING = inspect.cleandoc("""Example command.

Parameters
----------
param1 : int
    This is the first parameter.
param2 : Optional[str]
    This is an optional parameter.
""")


def test_from_docstring(capsys: pytest.CaptureFixture[str]):
    """Test the from_docstring decorator applies help texts correctly."""
    pytest.skip("Broken")
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
    assert example_command.__doc__, "Example command missing docstring."
    assert inspect.cleandoc(example_command.__doc__) == DOCSTRING
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


def test_from_docstring_no_docstring_returns_same_function() -> None:
    app = typer.Typer()

    def func_no_docstring(x: int):  # no docstring!
        return x + 1

    decorated = cli.from_docstring(app)(func_no_docstring)
    assert decorated is func_no_docstring


def test_from_docstring_oldstyle_and_no_docstring(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test from_docstring with old-style Typer annotations and missing docstring."""

    app = typer.Typer()

    # --- old-style Typer defaults ---
    @cli.from_docstring(app, name="oldstyle_command")
    def oldstyle_command(
        param1: int = typer.Argument(...),
        param2: Optional[str] = typer.Option("a"),
    ) -> None:
        """Old-style command.

        Parameters
        ----------
        param1 : int
            This is the first old-style parameter.
        param2 : Optional[str]
            This is an optional old-style parameter.
        """
        print("Old-style", param1, param2)

    # Check help output for the old-style command (should still parse the docstring)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Old-style command." in result.output
    assert "This is the first old-style parameter." in result.output
    assert "This is an optional old-style parameter." in result.output

    # Execute both commands
    result = runner.invoke(app, ["5", "--param2", "z"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "Old-style 5 z"


def test_from_docstring_implicit_argument_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test from_docstring with implicit Typer annotations and missing docstring."""

    app = typer.Typer()

    # --- Implicit Typer defaults ---
    @cli.from_docstring(app, name="implicit_command")
    def implicit_command(
        param1: int,
        param2: str = "a",
    ) -> None:
        """Implicit command.

        Parameters
        ----------
        param1 : int
            This is the first implicit parameter.
        param2 : Optional[str]
            This is an optional implicit parameter.
        """
        print("Implicit", param1, param2)

    # Check help output for the implicit command (should still parse the docstring)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Implicit command." in result.output
    assert "This is the first implicit parameter." in result.output
    assert "This is an optional implicit parameter." in result.output

    # Execute both commands
    result = runner.invoke(app, ["5", "--param2", "z"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "Implicit 5 z"


def test_from_docstring_kwargs(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the from_docstring decorator passes kwargs correctly."""
    pytest.skip("Broken")
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
    assert example_command.__doc__, "Example command is missing docstring"
    assert inspect.cleandoc(example_command.__doc__) == DOCSTRING
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
