import typer
from typer.testing import CliRunner
from typing import Annotated, Optional
import pytest

from qcore import cli
import pytest

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
