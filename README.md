<h1 align="center"><img src="asset/waypoint.svg" alt="alt text" width="500"></h1>
</br>


[![PyPI](https://img.shields.io/pypi/v/waypoint?label=PyPI)](https://pypi.org/project/waypoint/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Pyright](https://img.shields.io/badge/pyright-checked-blue)](https://github.com/microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://img.shields.io/github/actions/workflow/status/cswartzvi/waypoint/testing.yaml?branch=main&label=tests&logo=github)](https://github.com/cswartzvi/waypoint/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/cswartzvi/waypoint/graph/badge.svg?token=1o01x0xk7i)](https://codecov.io/github/cswartzvi/waypoint)

Waypoint is a lightweight alternative to cloud-native orchestrators, giving scientists and engineers the flexibility to design and run workflows with full control — no vendor lock-in, no central server required.

## Result storage

Waypoint ships with a simple `ResultStore` protocol for persisting task outputs. An
`FSSpecResultStore` implementation leverages [`fsspec`](https://filesystem-spec.readthedocs.io/) to read and write data to any supported filesystem.

Install the optional dependency:

```bash
pip install "waypoint[fsspec]"
```

```python
from waypoint.results import FSSpecResultStore

store = FSSpecResultStore()
store.write_bytes("output.bin", b"data")
assert store.exists("output.bin")
print(store.read_bytes("output.bin"))
```

You can also provide a custom filesystem:

```python
import fsspec

fs = fsspec.filesystem("memory")
store = FSSpecResultStore(fs)
```
