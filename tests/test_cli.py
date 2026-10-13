import inspect
from typing import Annotated, get_args

import pytest
import typer
from typer.models import ArgumentInfo, OptionInfo
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

    app = typer.Typer()

    @cli.from_docstring(app)
    def example_command(
        param1: Annotated[int, typer.Argument()],
        param2: Annotated[str | None, typer.Option()] = "a",
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


def test_from_docstring_oldstyle_and_no_docstring() -> None:
    """Test from_docstring with old-style Typer annotations and missing docstring."""

    app = typer.Typer()

    # --- old-style Typer defaults ---
    @cli.from_docstring(app, name="oldstyle_command")
    def oldstyle_command(
        param1: int = typer.Argument(...),
        param2: str | None = typer.Option("a"),
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


def test_from_docstring_implicit_argument_options() -> None:
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


def test_from_docstring_implicit_signature() -> None:
    """Check the rewritten signature for implicit arguments and options."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command(
        documented: int,
        undocumented,  # noqa: ANN001
        opt: float = 1.5,
        undoc_opt=2,  # noqa: ANN001
    ) -> None:
        """Summary.

        Parameters
        ----------
        documented : int
            Documented argument.
        opt : float
            Documented option.
        """

    params = inspect.signature(command).parameters

    documented = params["documented"]
    assert isinstance(documented.default, ArgumentInfo)
    assert documented.default.default is ...
    assert documented.default.help == "Documented argument."
    assert documented.annotation is int

    undocumented = params["undocumented"]
    assert isinstance(undocumented.default, ArgumentInfo)
    assert undocumented.default.help == ""
    assert undocumented.annotation is str

    opt = params["opt"]
    assert isinstance(opt.default, OptionInfo)
    assert opt.default.default == 1.5
    assert opt.default.help == "Documented option."
    assert opt.annotation is float

    undoc_opt = params["undoc_opt"]
    assert isinstance(undoc_opt.default, OptionInfo)
    assert undoc_opt.default.default == 2
    assert undoc_opt.default.help == ""
    assert undoc_opt.annotation is str


def test_from_docstring_implicit_runtime_behaviour() -> None:
    """Required arguments are required and option defaults are kept."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command(value: int, scale: float = 1.5) -> None:
        """Summary."""
        print(type(value).__name__, value, scale)

    result = runner.invoke(app, [])
    assert result.exit_code != 0

    result = runner.invoke(app, ["3"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "int 3 1.5"


def test_from_docstring_keeps_explicit_annotated_help() -> None:
    """Explicit help in Annotated metadata is not overwritten by the docstring."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command(
        explicit: Annotated[int, typer.Argument(help="Explicit help.")],
        implicit: Annotated[int, typer.Option()] = 1,
    ) -> None:
        """Summary.

        Parameters
        ----------
        explicit : int
            Docstring help.
        implicit : int
            Implicit help.
        """

    params = inspect.signature(command).parameters
    (explicit_info,) = get_args(params["explicit"].annotation)[1:]
    assert explicit_info.help == "Explicit help."
    (implicit_info,) = get_args(params["implicit"].annotation)[1:]
    assert implicit_info.help == "Implicit help."


def test_from_docstring_command_help_short_only() -> None:
    """A docstring without a long description yields just the summary."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command() -> None:
        """Summary only."""

    assert app.registered_commands[0].help == "Summary only."


def test_from_docstring_command_help_with_long_description() -> None:
    """The long description is appended to the summary."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command() -> None:
        """Summary.

        Longer description.
        """

    assert app.registered_commands[0].help == "Summary.\n\nLonger description."


def test_from_docstring_command_help_no_summary() -> None:
    """A docstring with only a parameters section gives empty command help."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command(x: int) -> None:
        """
        Parameters
        ----------
        x : int
            X help.
        """

    assert app.registered_commands[0].help == ""
    x = inspect.signature(command).parameters["x"]
    assert isinstance(x.default, ArgumentInfo)
    assert x.default.help == "X help."


def test_from_docstring_forces_numpydoc_style() -> None:
    """The docstring is parsed as numpydoc even if another style has more fields."""

    app = typer.Typer()

    @cli.from_docstring(app)
    def command(x: int) -> None:
        """Summary.

        :param a: Not a numpydoc parameter.
        :param b: Not a numpydoc parameter.

        Parameters
        ----------
        x : int
            X help.
        """

    x = inspect.signature(command).parameters["x"]
    assert isinstance(x.default, ArgumentInfo)
    assert x.default.help == "X help."
