# Patches

Drop-in framework patches required to reproduce HNSecW's 2PC results.
Apply them after cloning the upstream framework, before building.

## `aby_arith_reset_clear_mvC.patch`

A one-line fix to `ArithSharing<T>::Reset()` in
[ABY](https://github.com/encryptogroup/ABY): the loop that frees the
Beaver-triple buffers between calls misses `m_vC` (the C-component).
On a Reset-and-reuse path the next setup phase therefore re-Creates
`m_vC[0]` over a buffer that still holds the previous round's C
values, and `VerifyArithMT` then reports a wave of `a*b != c` failures
starting at the index where the second setup overlaps.  Apply with:

```
cd /path/to/ABY
git apply /path/to/Secure_ANN/patches/aby_arith_reset_clear_mvC.patch
```

`hnsecw_single_b2y.cpp` does not depend on this patch in steady state
(it allocates a fresh `ABYParty` per iteration, sidestepping the
missing free), but the patch makes `Reset()` self-consistent and
unblocks any future caller that wants to reuse the party.
