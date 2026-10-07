#!/usr/bin/env python3
"""Write the party-0 input file for Programs/Source/hnsecw_3pc.mpc and print
the result the program must output.

The layer is either synthetic (--n, --m, --d: random graph and vectors, which
is enough to measure cost) or one layer of an index built by
2pc/bench/build_index.py (--index, --layer, --queries).  run_search.py uses
this module to search all layers of an index.

Input file layout (int64 each):
    graph        (N+1) x (columns*M)   one-hot digits of the neighbour addresses
    vectors      (N+1) x D
    norms        (N+1)
    queries      Q x D
    entries      Q

The expected results come from a plaintext replay that follows the program
step by step, including its tie-breaking, so the two outputs must be identical.
"""

import argparse
import json
import math

import numpy as np


def layout(n, m, t, q, mode):
    """Address space of the program (see hnsecw_3pc.mpc)."""
    batch, multi = mode == "batch", mode == "multi"
    u = q if batch else 1
    k = u * m
    lanes = max(u, 2)
    vec_steps = q * (t + 1) if multi else t
    graph_steps = (q * t if multi else t) if (batch or multi) else 0
    vec_dum = n + (u if batch else 0)
    graph_dum = vec_dum + vec_steps * k
    c_sent = graph_dum + graph_steps * lanes
    return dict(u=u, k=k, lanes=lanes, vec_dum=vec_dum, graph_dum=graph_dum, c_sent=c_sent,
                sigma=c_sent + t)


def address_bits(n, m, t, q, mode):
    return max(4, layout(n, m, t, q, mode)["sigma"].bit_length())


def digit_widths(bits, digit):
    """Digits of an address, least significant first (as in the program)."""
    def cut(n_bits):
        n = -(-n_bits // digit)
        return [n_bits // n + 1] * (n_bits % n) + [n_bits // n] * (n - n_bits % n)

    return cut((bits + 1) // 2) + cut(bits // 2)


def graph_columns(bits, digit=8):
    """Shared values per neighbour slot in the graph table."""
    return sum(1 if w == 1 else 1 << w for w in digit_widths(bits, digit))


def digit_planes(graph, bits, digit):
    """The graph table as the program stores it: every address in digits,
    every digit one-hot (a one-bit digit as the bit itself).  Returns an array
    of shape (rows, columns * M); the column of value v of a digit holds, for
    every neighbour slot, whether the digit equals v."""
    planes = []
    offset = 0
    for w in digit_widths(bits, digit):
        value = (graph >> offset) & ((1 << w) - 1)
        if w == 1:
            planes.append(value)
        else:
            planes.extend((value == v).astype(np.int64) for v in range(1 << w))
        offset += w
    return np.concatenate(planes, axis=1)


class Reference:
    """Plaintext replay of the program.  Rows are (distkey, address); every
    sort is stable, which is what the position tie-break of the program does."""

    def __init__(self, graph, vecs, n, m, t, ef, vbits):
        self.graph, self.vecs, self.n, self.m, self.t, self.ef = graph, vecs, n, m, t, ef
        d = vecs.shape[1]
        self.dist_bits = 2 * vbits + max(1, math.ceil(math.log2(d)))
        self.sentinel = 2 ** (self.dist_bits + 1) - 1

    def row(self, addr):
        return addr if addr < self.n else self.n

    def scored(self, query, addr):
        """A dummy address reads the dummy row, whose norm is 2^dist_bits."""
        if addr >= self.n:
            return ((1 << self.dist_bits) + int(query @ query), addr)
        diff = self.vecs[addr] - query
        return (int(diff @ diff), addr)

    @staticmethod
    def best(rows, count):
        return sorted(rows, key=lambda r: r[0])[:count]

    def single(self, query, entry):
        lay = layout(self.n, self.m, self.t, 1, "single")
        visited = {lay["sigma"], entry}
        cand = [self.scored(query, entry)] + [(self.sentinel, lay["c_sent"] + j) for j in range(self.t - 1)]
        popped = []
        for t in range(self.t):
            head, rest = cand[0], cand[1:]
            popped.append(head)
            new = []
            for k in range(self.m):
                x = int(self.graph[self.row(head[1]), k])
                opened = lay["vec_dum"] + t * self.m + k if x in visited else x
                visited.add(opened)
                new.append(self.scored(query, opened))
            cand = self.best(rest + new, self.t)
        return [addr for _, addr in self.best(popped + rest + new, self.ef)]

    def multi(self, queries, entries):
        """Queries one after another with shared visited lists: a query first
        scores every vector fetched so far."""
        lay = layout(self.n, self.m, self.t, len(queries), "multi")
        seen_v, seen_g = {lay["sigma"]}, {lay["sigma"]}
        fetched = [lay["sigma"]]
        results = []
        for u, (query, entry) in enumerate(zip(queries, entries)):
            step = u * (self.t + 1)
            for k in range(self.m):                      # lane 0: entry, the rest: sigma
                x = entry if k == 0 else lay["sigma"]
                opened = lay["vec_dum"] + step * self.m + k if x in seen_v else x
                seen_v.add(opened)
                fetched.append(opened)
            sentinels = [(self.sentinel, lay["c_sent"] + r) for r in range(self.t)]
            cand = self.best(sentinels + [self.scored(query, a) for a in fetched], self.t)
            popped = []
            for t in range(self.t):
                head, rest = cand[0], cand[1:]
                popped.append(head)
                a = head[1]
                opened = lay["graph_dum"] + (u * self.t + t) * lay["lanes"] if a in seen_g else a
                seen_g.add(opened)
                row = self.row(opened)
                new = []
                for k in range(self.m):
                    x = int(self.graph[row, k])
                    opened = lay["vec_dum"] + (step + 1 + t) * self.m + k if x in seen_v else x
                    seen_v.add(opened)
                    fetched.append(opened)
                    new.append(self.scored(query, opened))
                cand = self.best(rest + new, self.t)
            results.append([addr for _, addr in self.best(popped + rest + new, self.ef)])
        return results

    def batch(self, queries, entries):
        u_count = len(queries)
        lay = layout(self.n, self.m, self.t, u_count, "batch")
        k_count = lay["k"]
        seen_v, seen_g = {lay["sigma"]}, {lay["sigma"]}
        new = []
        for u in range(u_count):
            opened = self.n + u if entries[u] in entries[:u] else entries[u]
            seen_v.add(opened)
            new.append(opened)
        sentinels = [(self.sentinel, lay["c_sent"] + r) for r in range(self.t)]
        cand = [self.best(sentinels + [self.scored(queries[u], o) for o in new], self.t)
                for u in range(u_count)]
        popped = [[] for _ in range(u_count)]
        for t in range(self.t):
            heads = [c[0] for c in cand]
            rest = [c[1:] for c in cand]
            rows = []
            for u in range(u_count):
                popped[u].append(heads[u])
                a = heads[u][1]
                opened = lay["graph_dum"] + t * u_count + u if a in seen_g else a
                seen_g.add(opened)
                rows.append(self.row(opened))
            new = []
            for u in range(u_count):
                for k in range(self.m):
                    x = int(self.graph[rows[u], k])
                    opened = lay["vec_dum"] + t * k_count + u * self.m + k if x in seen_v else x
                    seen_v.add(opened)
                    new.append(opened)
            scored = [[self.scored(queries[u], o) for o in new] for u in range(u_count)]
            cand = [self.best(rest[u] + scored[u], self.t) for u in range(u_count)]
        return [[addr for _, addr in self.best(popped[u] + rest[u] + scored[u], self.ef)]
                for u in range(u_count)]


def synthetic_layer(n, m, d, q, rng):
    """Random vectors, a random graph with distinct neighbours, random queries."""
    vecs = rng.integers(0, 256, size=(n, d), dtype=np.int64)
    graph = np.full((n, m), n, dtype=np.int64)
    for i in range(n):
        row = rng.choice(n - 1, size=min(m, n - 1), replace=False)
        row[row >= i] += 1
        graph[i, : len(row)] = row
    queries = rng.integers(0, 256, size=(q, d), dtype=np.int64)
    return graph, vecs, queries


def write_inputs(path, graph, vecs, queries, entries, t, ef, mode, bits=0, digit=8):
    """Writes the party-0 input file.  graph: (N, M) with unused slots >= N.
    Returns (address bits, expected results)."""
    graph = np.asarray(graph, dtype=np.int64)
    vecs = np.asarray(vecs, dtype=np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    n, d = vecs.shape
    m = graph.shape[1]
    q = queries.shape[0]
    assert len(entries) == q and all(0 <= e < n for e in entries)
    lay = layout(n, m, t, q, mode)
    bits = bits or address_bits(n, m, t, q, mode)
    assert lay["sigma"] < 2 ** bits, "address bits too small"

    # Unused neighbour slots and the dummy row hold sigma.
    graph = np.where(graph < n, graph, lay["sigma"])
    graph = np.vstack([graph, np.full((1, m), lay["sigma"], dtype=np.int64)])
    vecs = np.vstack([vecs, np.zeros((1, d), dtype=np.int64)])

    ref = Reference(graph, vecs, n, m, t, ef, 8)
    with open(path, "wb") as f:
        digit_planes(graph, bits, digit).astype("<i8").tofile(f)
        vecs.astype("<i8").tofile(f)
        norms = (vecs * vecs).sum(axis=1)
        norms[n] = 2 ** ref.dist_bits                               # dummy row
        norms.astype("<i8").tofile(f)
        queries.astype("<i8").tofile(f)
        np.array(entries, dtype="<i8").tofile(f)

    if mode == "batch":
        expected = ref.batch(queries, list(entries))
    elif mode == "multi":
        expected = ref.multi(queries, list(entries))
    else:
        expected = [ref.single(queries[i], entries[i]) for i in range(q)]
    return bits, expected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="e.g. Player-Data/Input-Binary-P0-0")
    ap.add_argument("--expect", required=True, help="file to write the expected results to (JSON)")
    ap.add_argument("--index", help=".npz from 2pc/bench/build_index.py")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--queries", help=".npy with uint8 query vectors (with --index)")
    ap.add_argument("--entries", help="comma-separated layer-local entry ids, one per query")
    ap.add_argument("--n", type=int, default=1000, help="synthetic: number of nodes")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--t", type=int, required=True, help="iterations")
    ap.add_argument("--ef", type=int, required=True)
    ap.add_argument("--q", type=int, default=1, help="number of queries")
    ap.add_argument("--mode", default="single", choices=["single", "multi", "batch"])
    ap.add_argument("--digit", type=int, default=8,
                    help="largest stored address digit in bits (HNSW_DIGIT of the program)")
    ap.add_argument("--bits", type=int, default=0, help="address bits; 0 = smallest that fits")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.index:
        data = np.load(args.index, allow_pickle=True)
        vecs = data[f"vecs_{args.layer}"]
        graph = data[f"neigh_{args.layer}"]
        queries = np.load(args.queries)[: args.q]
    else:
        graph, vecs, queries = synthetic_layer(args.n, args.m, args.d, args.q, rng)
    n, d = vecs.shape
    m = graph.shape[1]
    q = queries.shape[0]
    entries = ([int(x) for x in args.entries.split(",")] if args.entries
               else [int(x) for x in rng.integers(0, n, size=q)])

    bits, expected = write_inputs(args.out, graph, vecs, queries, entries, args.t, args.ef,
                                  args.mode, args.bits, args.digit)
    with open(args.expect, "w") as f:
        json.dump({"n": n, "m": m, "d": d, "t": args.t, "ef": args.ef, "q": q, "bits": bits,
                   "mode": args.mode, "entries": entries, "expected": expected}, f)
    print(f"N={n} M={m} D={d} T={args.t} ef={args.ef} Q={q} bits={bits} mode={args.mode}")


if __name__ == "__main__":
    main()
