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


def test(session: nox.Session) -> None:
    """Run testing suite in an isolated environment."""
    pyproject = nox.project.load_toml("pyproject.toml")

    # FIXME: For some reason `session.install(".", "--all-extras")` does not work
    extras = ",".join(pyproject["project"]["optional-dependencies"].keys())
    session.install(f".[{extras}]", "--reinstall-package", "hamilton-composer", silent=False)

    session.install(*nox.project.dependency_groups(pyproject, "test"))
    session.run("pytest", "--cov", *session.posargs)
