# Two-party programs (ABY)

`scripts/build_aby.sh` builds one binary, `hnsecw_cli`, from the sources in
`src/hnsecw/`.  One run of the binary searches one HNSW layer;
`bench/run_search.py` chains the layers of an index.

```
hnsecw_cli -r <0|1> -X <single|multi|batch> -m M -d D -c T -w ef -i address_bits
           [-q queries] [-b coordinate_bits] [-f entry_file] [-o result_file]
           [-a server_address] [-p port] [-g 1]
```

Start one process with `-r 0` and one with `-r 1`.  With the environment
variable `HNSECW_DATA_FILE` the layer comes from a file (format in
`hnsecw_harness.h`); without it the layer is random, which is enough to
measure cost.  The client prints the online and the setup cost of the run.

## How an iteration is computed

| file | content |
|---|---|
| `hnsecw_core.h` | comparator networks, their Yao evaluation, the visited check |
| `hnsecw_search.h` | the two circuits of an iteration and the single-query layer search |
| `hnsecw_harness.h` | layer tables, cost accounting, circuit helpers |
| `hnsecw_single_b2y.cpp`, `hnsecw_multi_b2y.cpp`, `hnsecw_batch_b2y.cpp` | the three programs |

Every iteration runs two circuits.

**Visited check (GMW).**  The M neighbour addresses of the popped node are
XOR-shared.  The addresses opened so far form a public set, so membership of
a secret address needs no comparison against secret values: with the one-hot
encodings of the two halves of the address, membership is an XOR of products
selected by the public set.  The cost is a few thousand AND gates per
address, independent of how long the visited list has become.  An address
that was seen is replaced by a fresh dummy address, then the addresses are
opened.

**Update (arithmetic sharing and Yao).**  Squared distances are computed on
arithmetic shares and converted to Yao.  Candidates are sorted and merged
into the candidate array C and the result array W with Batcher networks that
are pruned to the outputs still needed: C only keeps as many entries as there
are iterations left.  C and W stay secret-shared between iterations; only
the address at the head of C is opened.  W is opened after the last
iteration.

The batch program runs the visited check on the U*M neighbours of all
queries at once, removes repeated addresses inside the batch obliviously, and
lets every query score all fetched vectors (Algorithm 4).

The multi program searches the queries one after another on one shuffled
copy of the tables.  The lists of opened addresses are shared by all
queries, so no address is opened twice.  A query starts with a catch-up
update that scores every vector fetched so far (their addresses are public,
so the shares behind them can be read again) and keeps the best as its
initial C and W.  Its iterations then check the popped address against the
graph rows opened so far, and the neighbours against the shared visited
list, as the batch program does.  The catch-up grows with the number of
vectors fetched by earlier queries.

Every value a run opens is compared with a plaintext replay that uses the
same networks, and the run fails if they differ.
