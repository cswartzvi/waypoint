import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "checker,subcommand",
    [
        ("pyright", None),
        ("mypy", None),
        # ("pyrefly", "check"),
        # ("ty", "check"),  # Uncomment when ty is more stable
    ],
)
def test_typing(checker: str, subcommand: str) -> None:
    """Run the type checker on the typing test file and ensure there are no errors."""
    file = (Path(__file__).parent / "assert_types.py").as_posix()
    if subcommand:
        command = [checker, subcommand, file]
    else:
        command = [checker, file]
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0
