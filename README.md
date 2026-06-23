# TinyTorch Learning Lab

## Purpose
I created this folder to get to know more into the underlying principles of ML algorithms and Deeplearning.

## Learning approach
Study -> experiment -> implement -> verify -> document.

## Repository structure
- `modules/`: lesson notes, experiments, and notebooks for each topic.
- `src/tinytorch/`: the Python package where TinyTorch-style components are implemented.
- `examples/`: small scripts that demonstrate how the implementation is meant to be used.
- `tests/`: focused checks for behavior as each component is built.

## Current progress
- Module 01: tensors.
- See `progress.md` for the running checklist once milestones are added.

## Running the project
This project currently targets Python 3.13 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run examples from the repository root:

```powershell
python examples\basic_tensor_usage.py
```

Run tests:

```powershell
python -m pytest
```

## Source
This learning lab is inspired by the TinyTorch curriculum and uses it as a guide for building deep-learning foundations from first principles.

Link: [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html)

## Long-term direction
Deep-learning foundations, ML systems, and later quantitative ML.
