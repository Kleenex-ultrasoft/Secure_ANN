# Three-party program (MP-SPDZ)

`Programs/Source/hnsecw_3pc.mpc` runs with `replicated-ring-party.x`
(semi-honest replicated secret sharing over Z_2^64).  It must be compiled
with `-M -E ring -R 64`.  `bench/run_search.py` writes the input, compiles,
starts the three parties and checks the result; `bench/gen_inputs.py` writes
the input file of one layer and holds the plaintext replay.

Parameters are environment variables read at compile time; they are listed
at the top of the program.

`scripts/build_mpspdz.sh` builds the MP-SPDZ party that the driver runs.

## How an iteration is computed

1. **Pop.**  Row 0 of the candidate array C is popped and its graph address
   is opened.
2. **Visited check (binary shares).**  Membership of a secret address in the
   public list of opened addresses is an XOR of products of one-hot terms
   selected by that list, as in the two-party program.  The graph table
   stores every neighbour address as one-hot digits (arithmetic shares of
   bits, which convert to binary shares locally), so the one-hot encodings
   of the two address halves take at most one level of AND gates.  The
   products themselves are not shared: each party forms its summand of
   every product locally, adds the summands up as the list prescribes, and
   the address, or a fresh dummy address in place of a visited one, is
   opened from the sums.  For wide addresses the program forms only the
   one-hot terms the list needs.
3. **Distances.**  One matrix product per iteration; the table stores the
   squared norm of every vector.  The row that all dummy addresses map to has
   a norm above every real distance, so dummies sort last without further
   work.
4. **Merge.**  The M scored neighbours and the rest of C are shuffled, all
   pairs are compared, the outcomes are opened and the rows are put in rank
   order.  Keys are made distinct by appending the position of the row, so
   the outcomes are those of a uniformly random permutation.

The result array W is not maintained during the iterations: the ef best rows
are always among the popped rows and the rows merged by the last iteration,
so the last merge produces W.

### Prepared comparisons

The comparisons of a merge do not start from the keys.  Before the query,
party 0 chooses a random mask for every position of the shuffled array and
gives parties 1 and 2 shares of what they need to compare two masked keys:
tables for the difference of the two masks and products of random bits.
In the merge the shuffle opens "key + mask" for every row to parties 1 and
2, uniformly random values for each of them; the two then open masked
table entries to each other and the outcomes to everybody, two rounds in
all.  Party 0 never sees a masked key.

The prepared material depends neither on the query nor on the tables, and
most of it comes from seeds: party 0 sends party 2 one message of about
360 bits per comparison.  The driver reports it separately ("prepared").
`HNSW_PREPARE=0` runs standard comparisons instead: nothing is prepared and
a merge takes five rounds more.

### Rounds

One iteration of a single query takes 8 rounds: pop 1, visited check 2,
distances 1, shuffle 2, comparisons 2.  On a layer with addresses of at
most 16 bits the visited check takes one round.

## Several queries

*multi*: the queries are searched one after another on one shuffled copy.
The visited list of the vector table and the list of opened graph rows are
shared by all queries.  A query first looks up its entry point through the
visited check, then scores every vector fetched so far and selects the T
best as its initial C.  In its iterations the popped address stays shared
and goes through a check against the opened graph rows before the row is
read.  Selecting T rows out of many thousands uses pivots: every row is
compared with a few pivot rows of the shuffled array, which gives a shortlist
containing the T smallest, and the step is repeated on the shortlist (see
`Sorter` in the program).

*batch*: all queries advance together (Algorithm 4).  The popped addresses
of the queries are compared with each other and with the graph rows opened
so far; the U*M neighbours are checked against one visited list and against
each other; every query scores all fetched vectors.  The arrays of all
queries are shuffled in the same two rounds, and the merge uses one pivot
stage, with prepared comparisons for the pivots and for the shortlist.  The
material of an iteration is prepared between two iterations to keep it out
of memory until it is needed; it is still independent of the inputs and is
reported as "prepared".
