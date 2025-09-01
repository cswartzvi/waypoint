import subprocess

import pytest


@pytest.mark.parametrize(
    "checker,subcommand",
    [
        ("pyright", None),
        ("mypy", None),
        ("pyrefly", "check"),
        # ("ty", "check"),  # Uncomment when ty is more stable
    ],
)
def test_typing(checker: str, subcommand: str) -> None:
    """Run the type checker on the typing test file and ensure there are no errors."""
    if subcommand:
        command = [checker, subcommand, "tests/typing/assert_types.py"]
    else:
        command = [checker, "tests/typing/assert_types.py"]
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0
