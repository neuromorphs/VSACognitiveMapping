# VSACognitiveMapping

## Summary

TODO

## Members

- TODO

## Setup

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### PyTorch

PyTorch is not pinned in `pyproject.toml` because the correct build depends on your OS and hardware (CPU-only, CUDA, ROCm, etc.). Install it separately using the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the command that matches your system.

Install PyTorch *before* the `detection` extra (`pip install -e ".[notebook,detection]"`) — `ultralytics` depends on `torch`, and installing it afterwards will pull in a generic build rather than the one you picked for your hardware.

## Acknowledgments

TODO: acknowledge the Telluride Neuromorphic AI Workshop.
