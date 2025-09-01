# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "nox",
# ]
# ///

import nox  # pyright: ignore

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "test"]
# nox.options.envdir = ".cache"


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    """Run linters in an isolated environment."""
    session.install("pre-commit", "pre-commit-uv")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def test(session: nox.Session) -> None:
    """Run testing suite in an isolated environment."""
    session.run_install(
        "uv",
        "sync",
        "--dev",
        "--all-extras",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run(
        "pytest",
        "--cov=waypoint",
        "--cov-branch",
        "--cov-report=html",
        "--cov-report=term",
        *session.posargs,
    )
