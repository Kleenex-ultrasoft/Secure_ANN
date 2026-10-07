// Shared building blocks for the HNSecW two-party programs:
//   * comparator networks (top-k selection and truncated merge),
//   * a SIMD Yao evaluator for those networks,
//   * the visited check against the public list of opened addresses,
//   * share bookkeeping for the benchmark harness.
#ifndef HNSECW_CORE_H
#define HNSECW_CORE_H

#include "../../abycore/aby/abyparty.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/sharing/sharing.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hnsecw {

// ---------------------------------------------------------------------------
// Comparator networks
// ---------------------------------------------------------------------------

// After a comparator, slot `a` holds the smaller key and slot `b` the larger.
struct Cmp {
    uint32_t a;
    uint32_t b;
};

struct Network {
    uint32_t n = 0;              // number of input slots
    std::vector<Cmp> cmps;       // in execution order
    std::vector<uint32_t> outs;  // outs[r] = slot holding rank r on exit
};

namespace detail {

inline uint32_t NextPow2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Batcher odd-even merge on positions lo, lo+r, lo+2r, ... < lo+n.
inline void OddEvenMerge(uint32_t lo, uint32_t n, uint32_t r,
                         std::vector<std::pair<uint32_t, uint32_t>>& out) {
    uint32_t m = r * 2;
    if (m < n) {
        OddEvenMerge(lo, n, m, out);
        OddEvenMerge(lo + r, n, m, out);
        for (uint32_t i = lo + r; i + r < lo + n; i += m) {
            out.emplace_back(i, i + r);
        }
    } else {
        out.emplace_back(lo, lo + r);
    }
}

inline void OddEvenSort(uint32_t lo, uint32_t n,
                        std::vector<std::pair<uint32_t, uint32_t>>& out) {
    if (n > 1) {
        uint32_t m = n / 2;
        OddEvenSort(lo, m, out);
        OddEvenSort(lo + m, m, out);
        OddEvenMerge(lo, n, 1, out);
    }
}

// Positions that hold no element behave as +infinity, so every comparator
// touching one is resolved at construction time and costs no gate.
inline std::vector<uint32_t> Resolve(
    const std::vector<std::pair<uint32_t, uint32_t>>& pos_cmps,
    std::vector<int64_t> slot_at,
    uint32_t keep,
    std::vector<Cmp>& emit) {
    for (const auto& pc : pos_cmps) {
        int64_t a = slot_at[pc.first];
        int64_t b = slot_at[pc.second];
        if (b < 0) continue;
        if (a < 0) {
            slot_at[pc.first] = b;
            slot_at[pc.second] = -1;
            continue;
        }
        emit.push_back({(uint32_t)a, (uint32_t)b});
    }
    std::vector<uint32_t> outs;
    outs.reserve(keep);
    for (uint32_t p = 0; p < keep; p++) {
        assert(slot_at[p] >= 0);
        outs.push_back((uint32_t)slot_at[p]);
    }
    return outs;
}

inline std::vector<uint32_t> EmitMerge(const std::vector<uint32_t>& A,
                                       const std::vector<uint32_t>& B,
                                       uint32_t keep,
                                       std::vector<Cmp>& emit) {
    keep = std::min<uint32_t>(keep, (uint32_t)(A.size() + B.size()));
    if (A.empty()) return std::vector<uint32_t>(B.begin(), B.begin() + keep);
    if (B.empty()) return std::vector<uint32_t>(A.begin(), A.begin() + keep);
    uint32_t n = NextPow2((uint32_t)std::max(A.size(), B.size()));
    std::vector<int64_t> slot_at(2 * n, -1);
    for (size_t i = 0; i < A.size(); i++) slot_at[i] = A[i];
    for (size_t i = 0; i < B.size(); i++) slot_at[n + i] = B[i];
    std::vector<std::pair<uint32_t, uint32_t>> pc;
    OddEvenMerge(0, 2 * n, 1, pc);
    return Resolve(pc, slot_at, keep, emit);
}

inline std::vector<uint32_t> EmitSortTopK(const std::vector<uint32_t>& slots,
                                          uint32_t k,
                                          std::vector<Cmp>& emit) {
    uint32_t n = (uint32_t)slots.size();
    if (n <= 1) return slots;
    if (n <= k) {
        uint32_t p = NextPow2(n);
        std::vector<int64_t> slot_at(p, -1);
        for (uint32_t i = 0; i < n; i++) slot_at[i] = slots[i];
        std::vector<std::pair<uint32_t, uint32_t>> pc;
        OddEvenSort(0, p, pc);
        return Resolve(pc, slot_at, n, emit);
    }
    uint32_t h = n / 2;
    std::vector<uint32_t> L(slots.begin(), slots.begin() + h);
    std::vector<uint32_t> R(slots.begin() + h, slots.end());
    std::vector<uint32_t> sl = EmitSortTopK(L, k, emit);
    std::vector<uint32_t> sr = EmitSortTopK(R, k, emit);
    return EmitMerge(sl, sr, k, emit);
}

// Drop comparators whose outputs cannot reach a requested output.
inline void Prune(Network& net) {
    std::vector<uint8_t> live(net.n, 0);
    for (uint32_t s : net.outs) live[s] = 1;
    std::vector<Cmp> kept;
    kept.reserve(net.cmps.size());
    for (size_t i = net.cmps.size(); i-- > 0;) {
        const Cmp& c = net.cmps[i];
        if (live[c.a] || live[c.b]) {
            kept.push_back(c);
            live[c.a] = 1;
            live[c.b] = 1;
        }
    }
    std::reverse(kept.begin(), kept.end());
    net.cmps.swap(kept);
}

}  // namespace detail

// Sorted top-k of n unsorted elements (slots 0..n-1).
inline Network MakeSortTopK(uint32_t n, uint32_t k) {
    Network net;
    net.n = n;
    std::vector<uint32_t> slots(n);
    for (uint32_t i = 0; i < n; i++) slots[i] = i;
    net.outs = detail::EmitSortTopK(slots, k, net.cmps);
    if (net.outs.size() > k) net.outs.resize(k);
    detail::Prune(net);
    return net;
}

// Smallest `keep` of two sorted runs: slots 0..p-1 followed by slots p..p+q-1.
inline Network MakeMergeTopK(uint32_t p, uint32_t q, uint32_t keep) {
    Network net;
    net.n = p + q;
    std::vector<uint32_t> A(p), B(q);
    for (uint32_t i = 0; i < p; i++) A[i] = i;
    for (uint32_t i = 0; i < q; i++) B[i] = p + i;
    net.outs = detail::EmitMerge(A, B, keep, net.cmps);
    detail::Prune(net);
    return net;
}

// Reference evaluation with the same swap rule as the circuit (swap iff the
// key in slot a is strictly larger), so ties resolve identically.
template <class Elem>
std::vector<Elem> RunNetworkPlain(const Network& net, std::vector<Elem> v) {
    assert(v.size() == net.n);
    for (const Cmp& c : net.cmps) {
        if (v[c.a].first > v[c.b].first) std::swap(v[c.a], v[c.b]);
    }
    std::vector<Elem> out;
    out.reserve(net.outs.size());
    for (uint32_t s : net.outs) out.push_back(v[s]);
    return out;
}

// ---------------------------------------------------------------------------
// Bit-sliced arrays in Yao sharing
// ---------------------------------------------------------------------------

// `wires[l]` carries bit l of every element (nvals = n).  The first
// `key_bits` wires are the sort key, least significant bit first; the
// remaining wires are payload.
struct YArray {
    std::vector<uint32_t> wires;
    uint32_t n = 0;
};

inline YArray Gather(BooleanCircuit* yc, const YArray& in,
                     const std::vector<uint32_t>& pos) {
    YArray out;
    out.n = (uint32_t)pos.size();
    out.wires.resize(in.wires.size());
    bool identity = (pos.size() == in.n);
    for (size_t i = 0; identity && i < pos.size(); i++) identity = (pos[i] == i);
    if (identity) {
        out.wires = in.wires;
        return out;
    }
    std::vector<uint32_t> p(pos);
    for (size_t l = 0; l < in.wires.size(); l++) {
        out.wires[l] = yc->PutSubsetGate(in.wires[l], p.data(), out.n, true);
    }
    return out;
}

inline YArray Concat(BooleanCircuit* yc, const YArray& x, const YArray& y) {
    assert(x.wires.size() == y.wires.size());
    if (x.n == 0) return y;
    if (y.n == 0) return x;
    YArray out;
    out.n = x.n + y.n;
    out.wires.resize(x.wires.size());
    for (size_t l = 0; l < x.wires.size(); l++) {
        out.wires[l] = yc->PutCombinerGate(std::vector<uint32_t>{x.wires[l], y.wires[l]});
    }
    return out;
}

// Public per-element constants (bit l of values[i] on wire l, position i).
inline YArray ConstArray(BooleanCircuit* yc, const std::vector<uint32_t>& values,
                         uint32_t bits) {
    YArray out;
    out.n = (uint32_t)values.size();
    out.wires.resize(bits);
    uint32_t c[2] = {yc->PutConstantGate((UGATE_T)0, 1), yc->PutConstantGate((UGATE_T)1, 1)};
    std::vector<uint32_t> parts(values.size());
    for (uint32_t l = 0; l < bits; l++) {
        for (size_t i = 0; i < values.size(); i++) parts[i] = c[(values[i] >> l) & 1U];
        out.wires[l] = (parts.size() == 1) ? parts[0] : yc->PutCombinerGate(parts);
    }
    return out;
}

inline YArray SelectWires(const YArray& in, const std::vector<uint32_t>& idx) {
    YArray out;
    out.n = in.n;
    out.wires.reserve(idx.size());
    for (uint32_t i : idx) out.wires.push_back(in.wires[i]);
    return out;
}

// Record of one network application, enough to undo it (see UnapplyNetwork).
struct NetTrace {
    struct Level {
        uint32_t cnt = 0;                   // comparators per lane
        std::vector<uint32_t> pa, pb, pu;   // positions read from the previous layout
        uint32_t gt = 0;                    // swap decisions, nvals = lanes * cnt
        uint32_t n_prev = 0;                // width of the previous layout
    };
    std::vector<Level> levels;
    std::vector<uint32_t> out_pos;          // positions gathered from the last layout
    uint32_t n_last = 0;                    // width of the last layout
    uint32_t lanes = 1;
};

// Runs the network level by level.  All comparators of a level are evaluated
// as one SIMD comparison plus one SIMD conditional swap per wire.
//
// With lanes > 1 the input holds `lanes` independent arrays of net.n elements
// back to back (element s of lane j at position j * net.n + s) and the same
// network runs on each of them; the output is laid out the same way, in rank
// order within each lane.
inline YArray ApplyNetwork(BooleanCircuit* yc, const YArray& in, const Network& net,
                           uint32_t key_bits, uint32_t lanes = 1, NetTrace* trace = nullptr) {
    assert(in.n == net.n * lanes);
    const size_t nw = in.wires.size();

    std::vector<uint32_t> level_of(net.cmps.size());
    std::vector<uint32_t> depth(net.n, 0);
    uint32_t nlevels = 0;
    for (size_t i = 0; i < net.cmps.size(); i++) {
        const Cmp& c = net.cmps[i];
        uint32_t lv = std::max(depth[c.a], depth[c.b]);
        level_of[i] = lv;
        depth[c.a] = depth[c.b] = lv + 1;
        nlevels = std::max(nlevels, lv + 1);
    }
    std::vector<std::vector<uint32_t>> by_level(nlevels);
    for (size_t i = 0; i < net.cmps.size(); i++) by_level[level_of[i]].push_back((uint32_t)i);

    // last_use[s]: last level reading slot s; outputs stay live to the end.
    std::vector<int64_t> last_use(net.n, -1);
    for (size_t i = 0; i < net.cmps.size(); i++) {
        last_use[net.cmps[i].a] = std::max<int64_t>(last_use[net.cmps[i].a], level_of[i]);
        last_use[net.cmps[i].b] = std::max<int64_t>(last_use[net.cmps[i].b], level_of[i]);
    }
    for (uint32_t s : net.outs) last_use[s] = (int64_t)nlevels;

    // Slot s of lane j sits at base[s] + j * stride[s] in the current layout.
    std::vector<uint32_t> cur = in.wires;
    uint32_t cur_n = in.n;
    std::vector<int64_t> base(net.n), stride(net.n, net.n);
    for (uint32_t s = 0; s < net.n; s++) base[s] = s;

    if (trace) {
        trace->levels.clear();
        trace->lanes = lanes;
    }

    std::vector<uint32_t> pa, pb, pu, us;
    std::vector<uint8_t> in_level(net.n, 0);
    for (uint32_t lv = 0; lv < nlevels; lv++) {
        const std::vector<uint32_t>& ids = by_level[lv];
        const uint32_t cnt = (uint32_t)ids.size();
        for (uint32_t k = 0; k < cnt; k++) {
            const Cmp& c = net.cmps[ids[k]];
            in_level[c.a] = in_level[c.b] = 1;
        }
        us.clear();
        for (uint32_t s = 0; s < net.n; s++) {
            if (!in_level[s] && base[s] >= 0 && last_use[s] > (int64_t)lv) us.push_back(s);
        }
        const uint32_t nu = (uint32_t)us.size();
        pa.clear();
        pb.clear();
        pu.clear();
        for (uint32_t j = 0; j < lanes; j++) {
            for (uint32_t k = 0; k < cnt; k++) {
                const Cmp& c = net.cmps[ids[k]];
                pa.push_back((uint32_t)(base[c.a] + (int64_t)j * stride[c.a]));
                pb.push_back((uint32_t)(base[c.b] + (int64_t)j * stride[c.b]));
            }
        }
        for (uint32_t j = 0; j < lanes; j++) {
            for (uint32_t s : us) pu.push_back((uint32_t)(base[s] + (int64_t)j * stride[s]));
        }

        std::vector<uint32_t> wa(nw), wb(nw);
        for (size_t l = 0; l < nw; l++) {
            wa[l] = yc->PutSubsetGate(cur[l], pa.data(), (uint32_t)pa.size(), true);
            wb[l] = yc->PutSubsetGate(cur[l], pb.data(), (uint32_t)pb.size(), true);
        }
        std::vector<uint32_t> ka(wa.begin(), wa.begin() + key_bits);
        std::vector<uint32_t> kb(wb.begin(), wb.begin() + key_bits);
        uint32_t gt = yc->PutGTGate(ka, kb);

        std::vector<uint32_t> next(nw);
        for (size_t l = 0; l < nw; l++) {
            uint32_t d = yc->PutXORGate(wa[l], wb[l]);
            uint32_t sw = yc->PutANDGate(gt, d);
            std::vector<uint32_t> parts;
            parts.push_back(yc->PutXORGate(wa[l], sw));
            parts.push_back(yc->PutXORGate(wb[l], sw));
            if (nu > 0) {
                parts.push_back(yc->PutSubsetGate(cur[l], pu.data(), (uint32_t)pu.size(), true));
            }
            next[l] = yc->PutCombinerGate(parts);
        }
        cur.swap(next);

        if (trace) {
            NetTrace::Level tl;
            tl.cnt = cnt;
            tl.pa = pa;
            tl.pb = pb;
            tl.pu = pu;
            tl.gt = gt;
            tl.n_prev = cur_n;
            trace->levels.push_back(tl);
        }

        for (uint32_t s = 0; s < net.n; s++) base[s] = -1;
        for (uint32_t k = 0; k < cnt; k++) {
            const Cmp& c = net.cmps[ids[k]];
            base[c.a] = k;
            stride[c.a] = cnt;
            base[c.b] = (int64_t)lanes * cnt + k;
            stride[c.b] = cnt;
            in_level[c.a] = in_level[c.b] = 0;
        }
        for (uint32_t i = 0; i < nu; i++) {
            base[us[i]] = 2 * (int64_t)lanes * cnt + i;
            stride[us[i]] = nu;
        }
        cur_n = (2 * cnt + nu) * lanes;
    }

    YArray laid;
    laid.wires = cur;
    laid.n = cur_n;
    std::vector<uint32_t> pos;
    pos.reserve(net.outs.size() * lanes);
    for (uint32_t j = 0; j < lanes; j++) {
        for (uint32_t s : net.outs) {
            assert(base[s] >= 0);
            pos.push_back((uint32_t)(base[s] + (int64_t)j * stride[s]));
        }
    }
    if (trace) {
        trace->out_pos = pos;
        trace->n_last = cur_n;
    }
    return Gather(yc, laid, pos);
}

// Undoes a traced application of a full sorting network on a single wire:
// `wire` carries one bit per element in output (rank) order and the result
// carries the same bits in input order.  A conditional swap is its own
// inverse, so replaying the recorded swap decisions backwards costs one AND
// per comparator.  The network must keep every slot (net.outs covers all).
inline uint32_t UnapplyNetwork(BooleanCircuit* yc, uint32_t wire, const NetTrace& trace) {
    std::vector<uint32_t> inv(trace.n_last, 0);
    assert(trace.out_pos.size() == trace.n_last);
    for (uint32_t r = 0; r < trace.out_pos.size(); r++) inv[trace.out_pos[r]] = r;
    uint32_t cur = yc->PutSubsetGate(wire, inv.data(), trace.n_last, true);

    for (size_t li = trace.levels.size(); li-- > 0;) {
        const NetTrace::Level& lv = trace.levels[li];
        const uint32_t na = (uint32_t)lv.pa.size();
        const uint32_t nu = (uint32_t)lv.pu.size();
        assert(2 * na + nu == lv.n_prev);
        std::vector<uint32_t> ia(na), ib(na), iu(nu);
        for (uint32_t i = 0; i < na; i++) {
            ia[i] = i;
            ib[i] = na + i;
        }
        for (uint32_t i = 0; i < nu; i++) iu[i] = 2 * na + i;
        uint32_t a = yc->PutSubsetGate(cur, ia.data(), na, true);
        uint32_t b = yc->PutSubsetGate(cur, ib.data(), na, true);
        uint32_t d = yc->PutXORGate(a, b);
        uint32_t sw = yc->PutANDGate(lv.gt, d);
        std::vector<uint32_t> parts;
        parts.push_back(yc->PutXORGate(a, sw));
        parts.push_back(yc->PutXORGate(b, sw));
        if (nu > 0) parts.push_back(yc->PutSubsetGate(cur, iu.data(), nu, true));
        uint32_t comb = yc->PutCombinerGate(parts);

        std::vector<uint32_t> back(lv.n_prev, 0);
        for (uint32_t i = 0; i < na; i++) {
            back[lv.pa[i]] = i;
            back[lv.pb[i]] = na + i;
        }
        for (uint32_t i = 0; i < nu; i++) back[lv.pu[i]] = 2 * na + i;
        cur = yc->PutSubsetGate(comb, back.data(), lv.n_prev, true);
    }
    return cur;
}

// ---------------------------------------------------------------------------
// Visited check: membership of secret-shared addresses in a public set
// ---------------------------------------------------------------------------
//
// Every address the servers open is public from then on, so the visited list
// of a query is a public set S.  Whether a secret address x lies in S is the
// XOR, over s in S, of the product of the literals of x that spell s; at most
// one product is 1 because S has no repeated element.  Splitting the address
// bits into two halves x = (hi, lo) lets the products be shared:
//
//     [x in S] = XOR_h  onehot_hi[h] AND ( XOR_{(h,l) in S} onehot_lo[l] )
//
// so the cost is one AND per distinct half value plus the one-hot terms that
// are actually used.  The one-hot terms are built by a balanced tree, either
// on demand (one AND per term) or as a full table whose last row and column
// come for free from the rows summing to the parent term.  The AND count is
// therefore bounded by a constant in the number of address bits and does not
// grow with |S| once S is large.

class PublicSetMembership {
public:
    PublicSetMembership(BooleanCircuit* bc, const std::vector<uint32_t>& bits)
        : bc_(bc), bits_(bits) {}

    // `set` must not contain repeated values.  Returns a 1-bit wire with the
    // same nvals as the address bits.
    uint32_t Build(const std::vector<uint32_t>& set, uint32_t nvals) {
        const uint32_t k = (uint32_t)bits_.size();
        if (set.empty()) return bc_->PutConstantGate((UGATE_T)0, nvals);
        if (k == 1) {
            bool has0 = false, has1 = false;
            for (uint32_t s : set) (s & 1U) ? has1 = true : has0 = true;
            if (has0 && has1) return bc_->PutConstantGate((UGATE_T)1, nvals);
            return has1 ? bits_[0] : bc_->PutINVGate(bits_[0]);
        }

        // Pick the root split with the fewest AND gates.
        uint32_t best_split = k / 2;
        uint64_t best_cost = UINT64_MAX;
        for (uint32_t split = 1; split < k; split++) {
            std::vector<uint32_t> lo, hi;
            Project(set, 0, split, lo);
            Project(set, split, k, hi);
            uint64_t cost = std::min(lo.size(), hi.size()) +
                            Cost(0, split, lo) + Cost(split, k, hi);
            if (cost < best_cost) {
                best_cost = cost;
                best_split = split;
            }
        }
        and_gates_ = best_cost;

        const uint32_t split = best_split;
        std::vector<uint32_t> lo, hi;
        Project(set, 0, split, lo);
        Project(set, split, k, hi);
        Node* nlo = MakeNode(0, split, lo);
        Node* nhi = MakeNode(split, k, hi);

        const uint32_t lo_mask = (split >= 32) ? 0xFFFFFFFFu : ((1U << split) - 1U);
        const bool group_by_hi = hi.size() <= lo.size();
        std::unordered_map<uint32_t, std::vector<uint32_t>> groups;
        for (uint32_t s : set) {
            uint32_t l = s & lo_mask;
            uint32_t h = s >> split;
            if (group_by_hi) groups[h].push_back(l);
            else groups[l].push_back(h);
        }
        std::vector<uint32_t> keys;
        keys.reserve(groups.size());
        for (const auto& g : groups) keys.push_back(g.first);
        std::sort(keys.begin(), keys.end());

        uint32_t acc = 0;
        bool have = false;
        for (uint32_t key : keys) {
            std::vector<uint32_t>& members = groups[key];
            std::sort(members.begin(), members.end());
            Node* outer = group_by_hi ? nhi : nlo;
            Node* inner = group_by_hi ? nlo : nhi;
            uint32_t x = Term(inner, members[0]);
            for (size_t i = 1; i < members.size(); i++) {
                x = bc_->PutXORGate(x, Term(inner, members[i]));
            }
            uint32_t t = bc_->PutANDGate(Term(outer, key), x);
            acc = have ? bc_->PutXORGate(acc, t) : t;
            have = true;
        }
        for (Node* n : nodes_) delete n;
        nodes_.clear();
        return acc;
    }

    uint64_t and_gates() const { return and_gates_; }

private:
    struct Node {
        uint32_t lo = 0, hi = 0;
        bool dense = false;
        Node* left = nullptr;
        Node* right = nullptr;
        std::vector<uint32_t> table;                   // dense: all 2^k terms
        std::unordered_map<uint32_t, uint32_t> memo;   // lazy: used terms
        uint32_t inv = 0;                              // k == 1: NOT bit
        bool has_inv = false;
    };

    static void Project(const std::vector<uint32_t>& set, uint32_t lo, uint32_t hi,
                        std::vector<uint32_t>& out) {
        const uint32_t width = hi - lo;
        const uint32_t mask = (width >= 32) ? 0xFFFFFFFFu : ((1U << width) - 1U);
        out.clear();
        out.reserve(set.size());
        for (uint32_t s : set) out.push_back((s >> lo) & mask);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    static uint64_t DenseCost(uint32_t k) {
        return (k >= 32) ? UINT64_MAX : ((1ULL << k) - k - 1ULL);
    }

    // `vals` are the distinct values needed from the node, relative to its
    // own bit range.
    static uint64_t Cost(uint32_t lo, uint32_t hi, const std::vector<uint32_t>& vals) {
        const uint32_t k = hi - lo;
        if (k <= 1) return 0;
        const uint32_t mid = k / 2;
        std::vector<uint32_t> l, h;
        Project(vals, 0, mid, l);
        Project(vals, mid, k, h);
        uint64_t lazy = vals.size() + Cost(lo, lo + mid, l) + Cost(lo + mid, hi, h);
        return std::min(lazy, DenseCost(k));
    }

    Node* MakeNode(uint32_t lo, uint32_t hi, const std::vector<uint32_t>& vals,
                   bool force_dense = false) {
        Node* n = new Node();
        nodes_.push_back(n);
        n->lo = lo;
        n->hi = hi;
        const uint32_t k = hi - lo;
        if (k <= 1) return n;
        const uint32_t mid = k / 2;
        std::vector<uint32_t> l, h;
        Project(vals, 0, mid, l);
        Project(vals, mid, k, h);
        uint64_t lazy = vals.size() + Cost(lo, lo + mid, l) + Cost(lo + mid, hi, h);
        n->dense = force_dense || (DenseCost(k) <= lazy);
        n->left = MakeNode(lo, lo + mid, l, n->dense);
        n->right = MakeNode(lo + mid, hi, h, n->dense);
        return n;
    }

    void BuildTable(Node* n) {
        if (!n->table.empty()) return;
        const uint32_t k = n->hi - n->lo;
        if (k == 1) {
            n->table = {bc_->PutINVGate(bits_[n->lo]), bits_[n->lo]};
            return;
        }
        BuildTable(n->left);
        BuildTable(n->right);
        const std::vector<uint32_t>& L = n->left->table;
        const std::vector<uint32_t>& H = n->right->table;
        const uint32_t nl = (uint32_t)L.size();
        const uint32_t nh = (uint32_t)H.size();
        const uint32_t shift = n->left->hi - n->left->lo;
        n->table.assign((size_t)nl * nh, 0);
        std::vector<uint32_t> col(nl, 0);
        std::vector<uint8_t> col_set(nl, 0);
        for (uint32_t h = 0; h + 1 < nh; h++) {
            uint32_t row = 0;
            bool row_set = false;
            for (uint32_t l = 0; l + 1 < nl; l++) {
                uint32_t w = bc_->PutANDGate(L[l], H[h]);
                n->table[(h << shift) | l] = w;
                row = row_set ? bc_->PutXORGate(row, w) : w;
                row_set = true;
            }
            // Row h sums to H[h].
            uint32_t last = row_set ? bc_->PutXORGate(H[h], row) : H[h];
            n->table[(h << shift) | (nl - 1)] = last;
            for (uint32_t l = 0; l < nl; l++) {
                uint32_t w = n->table[(h << shift) | l];
                col[l] = col_set[l] ? bc_->PutXORGate(col[l], w) : w;
                col_set[l] = 1;
            }
        }
        // Column l sums to L[l].
        for (uint32_t l = 0; l < nl; l++) {
            n->table[((nh - 1) << shift) | l] =
                col_set[l] ? bc_->PutXORGate(L[l], col[l]) : L[l];
        }
    }

    uint32_t Term(Node* n, uint32_t v) {
        const uint32_t k = n->hi - n->lo;
        if (k == 1) {
            if (v & 1U) return bits_[n->lo];
            if (!n->has_inv) {
                n->inv = bc_->PutINVGate(bits_[n->lo]);
                n->has_inv = true;
            }
            return n->inv;
        }
        if (n->dense) {
            BuildTable(n);
            return n->table[v];
        }
        auto it = n->memo.find(v);
        if (it != n->memo.end()) return it->second;
        const uint32_t shift = n->left->hi - n->left->lo;
        const uint32_t lmask = (1U << shift) - 1U;
        uint32_t w = bc_->PutANDGate(Term(n->left, v & lmask), Term(n->right, v >> shift));
        n->memo.emplace(v, w);
        return w;
    }

    BooleanCircuit* bc_;
    std::vector<uint32_t> bits_;
    std::vector<Node*> nodes_;
    uint64_t and_gates_ = 0;
};

// ---------------------------------------------------------------------------
// Share bookkeeping for the benchmark harness
// ---------------------------------------------------------------------------
//
// The protocol runs over tables that the two servers hold in secret-shared
// form.  The harness loads the plaintext tables in both processes and splits
// every value it feeds to the circuit with a mask stream that both processes
// derive in the same order, so each process only ever inputs its own share
// and the circuits are exactly the ones a deployment with pre-shared tables
// would run.

class ShareStream {
public:
    ShareStream(e_role role, uint64_t seed) : role_(role), state_(seed) {}

    uint32_t Xor(uint32_t value, uint32_t bits) {
        uint32_t m = Mask(bits);
        return (role_ == SERVER) ? m : ((value ^ m) & Low(bits));
    }

    uint32_t Add(uint32_t value, uint32_t bits) {
        uint32_t m = Mask(bits);
        return (role_ == SERVER) ? m : ((value - m) & Low(bits));
    }

private:
    static uint32_t Low(uint32_t bits) {
        return (bits >= 32) ? 0xFFFFFFFFu : ((1U << bits) - 1U);
    }

    uint32_t Mask(uint32_t bits) {
        state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return (uint32_t)z & Low(bits);
    }

    e_role role_;
    uint64_t state_;
};

}  // namespace hnsecw

#endif  // HNSECW_CORE_H
