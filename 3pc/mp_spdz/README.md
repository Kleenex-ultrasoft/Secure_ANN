# MP-SPDZ (3PC)

This directory contains MP-SPDZ programs and utilities for HNSecW's 3PC
implementation. For online search, see `3pc/aby3`.

## Files

- `Programs/Source/` MP-SPDZ program sources.
- `gen_config.py` build a JSON config from an HNSW NPZ.
- `gen_inputs.py` generate MP-SPDZ input files.

## Build MP-SPDZ

```bash
cd /path/to/MP-SPDZ
make -j"$(nproc)" tldr
```

If you only need replicated ring:

```bash
make -j"$(nproc)" replicated-ring-party.x
```
