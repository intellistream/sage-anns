# sage-anns Copilot Instructions

## Scope
- Package: `isage-anns`, import path `sage_anns`.
- Layer: **L3** — C++/Python ANNS (Approximate Nearest Neighbor Search) library; no L4+ dependencies.
- Purpose: Unified Python/C++ interface for multiple ANNS algorithms (FAISS, DiskANN, HNSW/vsag, CANDY, PUCK, etc.).

## Polyrepo Context (Important)
SAGE was restructured from a monorepo into a polyrepo. `sage-anns` is a **standalone L3 C++ extension repo** providing ANNS backends. It integrates with `sage-libs` via the `sage.libs.ann` interface layer.

## Critical rules
- C++ extension built via `scikit-build-core` + pybind11; do not break the CMakeLists.txt build.
- Keep Python bindings in `python/sage_anns/`; C++ implementations under `implementations/`.
- Do not create new local virtual environments (`venv`/`.venv`); use the existing configured Python environment.
- In conda environments, use `python -m pip` (never plain `pip`).
- No fallback logic; fail fast.

## Build
```bash
./quickstart.sh       # installs hooks + builds C++ extension
./build_all.sh        # full rebuild of all ANNS implementations
pip install -e .      # editable install (triggers CMake build)
```

## Architecture focus
- `implementations/` — C++ ANNS implementations (FAISS, DiskANN, HNSW via vsag, CANDY, PUCK, SPTAG, etc.).
- `python/sage_anns/` — Python bindings + unified interface (`base.py`, `factory.py`, `algorithms/`, `wrappers/`).
- `include/` — shared C++ headers.
- `examples/` — usage examples.
- `tests/` — unit and integration tests.

## Dependencies
- **Depends on**: pybind11, PyTorch (CPU ABI-compatible build), `isage-libs` (L3 ann interfaces).
- **Depended on by**: `sage-libs`, `sage-middleware`, vector search application repos.

## Workflow
1. Make minimal changes; verify C++ builds cleanly before committing.
2. Keep Python API stable in `python/sage_anns/`.
3. Run `pytest tests/ -v` after any Python change.
4. For C++ changes, rebuild with `pip install -e .` and run tests.

## Git Hooks (Mandatory)
- Never use `git commit --no-verify` or `git push --no-verify`.
- If hooks fail, fix the issue first.
- Run `./quickstart.sh` after cloning to install hooks.
