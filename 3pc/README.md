# 3PC components

This directory contains the 3PC implementation of HNSecW, built on MP-SPDZ
with ABY3 integration for efficient online search.

## Architecture

- `mp_spdz/`: primary framework for offline reshuffle and a reference search
  implementation using replicated ring sharing.
- `aby3/`: online search module optimized for lower query latency with mixed
  arithmetic/Boolean sharing.

## Rationale

The paper describes our 3PC protocol using MP-SPDZ primitives. We integrate
ABY3 for the online query phase because its mixed-sharing model provides
faster comparison and conversion circuits than pure arithmetic protocols.

See each subdirectory for build and run instructions.
