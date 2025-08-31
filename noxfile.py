# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "nox",
# ]
# ///

import nox  # type: ignore

nox.options.default_venv_backend = "uv"
# nox.options.envdir = ".cache"


@nox.session()
def test(session: nox.Session) -> None:
    """Run testing suite in an isolated environment."""
    pyproject = nox.project.load_toml("pyproject.toml")

    # FIXME: For some reason `session.install(".", "--all-extras")` does not work
    extras = ",".join(pyproject["project"]["optional-dependencies"].keys())
    session.install(f".[{extras}]", "--reinstall-package", "hamilton-composer", silent=False)

    session.install(*nox.project.dependency_groups(pyproject, "test"))
    session.run("pytest", "--cov", *session.posargs)
