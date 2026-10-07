// The two circuits every HNSecW iteration is made of.
//
//   AccessCheck   decides which secret addresses may be opened: an address
//                 that was opened before (or that repeats inside the same
//                 batch) is replaced by a fresh dummy address, so no table
//                 row is ever read twice.
//
//   Update        scores the fetched vectors against the query, sorts them,
//                 and merges them into the candidate array C and the result
//                 array W.
//
// The single, multi and batch programs differ only in how they chain the two.
#ifndef HNSECW_SEARCH_H
#define HNSECW_SEARCH_H

#include "hnsecw_harness.h"

namespace hnsecw {

struct Params {
    uint32_t M = 0;
    uint32_t D = 0;
    uint32_t T = 0;        // iterations per layer
    uint32_t ef = 0;       // size of W
    uint32_t id_bits = 0;
    uint32_t key_bits = 0;
    uint32_t arith_bits = 32;
    uint32_t dmax = 0;     // all-ones key
};

// Returns false (after printing why) if the parameters are not supported.
inline bool InitParams(Params& p, uint32_t M, uint32_t D, uint32_t T, uint32_t ef,
                       uint32_t bitlen, uint32_t id_bitlen) {
    p.M = M;
    p.D = D;
    p.T = T;
    p.ef = ef;
    p.id_bits = id_bitlen;
    p.key_bits = 2 * bitlen + CeilLog2(D) + 1;
    if (p.key_bits > 31 || id_bitlen > 31 || id_bitlen == 0) {
        std::cerr << "distance or address width above 31 bits is not supported "
                  << "(distance bits=" << p.key_bits << ", id_bitlen=" << id_bitlen << ")" << std::endl;
        return false;
    }
    if (T == 0 || ef == 0 || M == 0 || D == 0 || ef > T) {
        std::cerr << "need M, D > 0 and 1 <= L_W <= L_C" << std::endl;
        return false;
    }
    p.dmax = (1U << p.key_bits) - 1U;
    return true;
}

// ---------------------------------------------------------------------------
// AccessCheck
// ---------------------------------------------------------------------------

enum class Dedup {
    kNone,      // the addresses are known to be distinct
    kPairwise,  // all pairs, GMW; for a handful of addresses
    kSort,      // sort and compare neighbours, Yao; for a large pool
};

struct AccessOut {
    std::vector<uint32_t> opened;      // addr[i], or dummy[i] if addr[i] was rejected
    std::vector<uint32_t> flag_share;  // this party's XOR share of "rejected"
    Cost cost;
};

// addr_share / dummy_share: this party's XOR shares of n addresses each.
// opened_before: every address of this table opened so far (no repeats).
inline AccessOut RunAccessCheck(ABYParty* party, uint32_t id_bits,
                                std::vector<uint32_t>& addr_share,
                                std::vector<uint32_t>& dummy_share,
                                const std::vector<uint32_t>& opened_before,
                                Dedup dedup) {
    const uint32_t n = (uint32_t)addr_share.size();
    party->Reset();
    std::vector<Sharing*>& sharings = party->GetSharings();
    BooleanCircuit* bc = (BooleanCircuit*)sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yc = (BooleanCircuit*)sharings[S_YAO]->GetCircuitBuildRoutine();
    ShareBin bin;

    share* id = bin.Keep(bc->PutSharedSIMDINGate(n, addr_share.data(), id_bits));
    share* dm = bin.Keep(bc->PutSharedSIMDINGate(n, dummy_share.data(), id_bits));

    PublicSetMembership member(bc, id->get_wires());
    uint32_t flag = member.Build(opened_before, n);

    if (dedup == Dedup::kPairwise && n > 1) {
        // dup[v] = OR over k >= 1 of [addr[v] == addr[v - k]]
        std::vector<uint32_t> eqs;
        for (uint32_t k = 1; k < n; k++) {
            std::vector<uint32_t> hi = Range(k, n);
            std::vector<uint32_t> lo = Range(0, n - k);
            std::vector<uint32_t> diff(id_bits);
            for (uint32_t l = 0; l < id_bits; l++) {
                uint32_t a = bc->PutSubsetGate(id->get_wire_id(l), hi.data(), n - k, true);
                uint32_t b = bc->PutSubsetGate(id->get_wire_id(l), lo.data(), n - k, true);
                diff[l] = bc->PutINVGate(bc->PutXORGate(a, b));  // 1 where the bits agree
            }
            while (diff.size() > 1) {
                std::vector<uint32_t> next;
                for (size_t i = 0; i + 1 < diff.size(); i += 2) next.push_back(bc->PutANDGate(diff[i], diff[i + 1]));
                if (diff.size() % 2 == 1) next.push_back(diff.back());
                diff.swap(next);
            }
            // positions v < k have no partner at distance k
            uint32_t pad = bc->PutConstantGate((UGATE_T)0, k);
            eqs.push_back(bc->PutCombinerGate(std::vector<uint32_t>{pad, diff[0]}));
        }
        while (eqs.size() > 1) {
            std::vector<uint32_t> next;
            for (size_t i = 0; i + 1 < eqs.size(); i += 2) next.push_back(bc->PutORGate(eqs[i], eqs[i + 1]));
            if (eqs.size() % 2 == 1) next.push_back(eqs.back());
            eqs.swap(next);
        }
        flag = bc->PutORGate(flag, eqs[0]);
    } else if (dedup == Dedup::kSort && n > 1) {
        // Sort the addresses, mark every element equal to its predecessor,
        // then send the marks back to the original order.
        share* idy = bin.Keep(yc->PutB2YGate(id));
        YArray arr;
        arr.n = n;
        arr.wires = idy->get_wires();
        Network net = MakeSortTopK(n, n);
        NetTrace trace;
        YArray sorted = ApplyNetwork(yc, arr, net, id_bits, 1, &trace);
        std::vector<uint32_t> hi = Range(1, n);
        std::vector<uint32_t> lo = Range(0, n - 1);
        uint32_t differ = 0;
        for (uint32_t l = 0; l < id_bits; l++) {
            uint32_t a = yc->PutSubsetGate(sorted.wires[l], hi.data(), n - 1, true);
            uint32_t b = yc->PutSubsetGate(sorted.wires[l], lo.data(), n - 1, true);
            uint32_t x = yc->PutXORGate(a, b);
            differ = (l == 0) ? x : yc->PutORGate(differ, x);
        }
        uint32_t first = yc->PutConstantGate((UGATE_T)0, 1);
        uint32_t dup_sorted = yc->PutCombinerGate(std::vector<uint32_t>{first, yc->PutINVGate(differ)});
        uint32_t dup = UnapplyNetwork(yc, dup_sorted, trace);
        share* dup_b = bin.Keep(bc->PutY2BGate(bin.Keep(new boolshare(std::vector<uint32_t>{dup}, yc))));
        flag = bc->PutORGate(flag, dup_b->get_wire_id(0));
    }

    // opened = flag ? dummy : addr
    std::vector<uint32_t> sel(id_bits);
    for (uint32_t l = 0; l < id_bits; l++) {
        uint32_t d = bc->PutXORGate(id->get_wire_id(l), dm->get_wire_id(l));
        uint32_t s = bc->PutANDGate(flag, d);
        sel[l] = bc->PutXORGate(id->get_wire_id(l), s);
    }
    share* open_out = bin.Keep(bc->PutOUTGate(bin.Keep(new boolshare(sel, bc)), ALL));
    share* flag_out = bin.Keep(bc->PutSharedOUTGate(bin.Keep(new boolshare(std::vector<uint32_t>{flag}, bc))));

    AccessOut res;
    res.cost = Execute(party);
    res.opened = ReadSimd(open_out, n);
    res.flag_share = ReadSimd(flag_out, n);
    return res;
}

// What the circuit above computes, on plaintext values.  With kSort the
// element that survives among equal addresses is the one the sorting network
// places first, so the replay runs the same network.
inline void PlainAccessCheck(const std::vector<uint32_t>& addr, const std::vector<uint32_t>& dummy,
                             const std::unordered_set<uint32_t>& opened_before, Dedup dedup,
                             std::vector<uint32_t>& opened, std::vector<uint8_t>& flag) {
    const size_t n = addr.size();
    opened.resize(n);
    flag.assign(n, 0);
    for (size_t i = 0; i < n; i++) flag[i] = opened_before.count(addr[i]) ? 1 : 0;
    if (dedup == Dedup::kPairwise) {
        std::unordered_set<uint32_t> seen;
        for (size_t i = 0; i < n; i++) {
            if (!seen.insert(addr[i]).second) flag[i] = 1;
        }
    } else if (dedup == Dedup::kSort && n > 1) {
        std::vector<std::pair<uint32_t, uint32_t>> items(n);
        for (size_t i = 0; i < n; i++) items[i] = std::make_pair(addr[i], (uint32_t)i);
        std::vector<std::pair<uint32_t, uint32_t>> sorted =
            RunNetworkPlain(MakeSortTopK((uint32_t)n, (uint32_t)n), items);
        for (size_t i = 1; i < n; i++) {
            if (sorted[i].first == sorted[i - 1].first) flag[sorted[i].second] = 1;
        }
    }
    for (size_t i = 0; i < n; i++) opened[i] = flag[i] ? dummy[i] : addr[i];
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

// Sizes and comparator networks of one update.  Everything here is a function
// of the public parameters and of how many nodes are still to be popped.
//
// After an update with `pops_left` pops still to come (the one that follows
// immediately included), C never needs more than pops_left entries: the
// entries beyond cannot be reached.  The array therefore shrinks by one per
// iteration instead of staying at T.
struct Plan {
    bool last = false;     // nothing is popped after this update
    uint32_t K = 0;        // candidates scored by every query
    uint32_t c_in = 0;     // entries of C carried into the update
    uint32_t c_keep = 0;   // entries of C after the merge, head included
    uint32_t kc = 0;       // sorted candidates offered to C
    uint32_t kw = 0;       // sorted candidates offered to W
    Network sort_cand;     // top max(kc, kw) of the K candidates
    Network merge_c;       // c_in + kc -> c_keep
    Network merge_w;       // ef + kw -> ef
};

inline Plan MakePlan(const Params& p, uint32_t K, uint32_t pops_left, uint32_t c_in) {
    Plan pl;
    pl.last = (pops_left == 0);
    pl.K = K;
    pl.c_in = c_in;
    pl.kc = std::min(K, pops_left);
    pl.kw = std::min(K, p.ef);
    pl.c_keep = std::min(pops_left, c_in + pl.kc);
    pl.sort_cand = MakeSortTopK(K, std::max(pl.kc, pl.kw));
    pl.merge_c = MakeMergeTopK(pl.c_in, pl.kc, pl.c_keep);
    pl.merge_w = MakeMergeTopK(p.ef, pl.kw, p.ef);
    return pl;
}

// This party's shares of the secret state carried between updates, for
// `lanes` queries; query j occupies positions [j * size, (j + 1) * size).
struct State {
    std::vector<uint32_t> c_key, c_pay;  // lanes * Plan::c_in entries, sorted by key per query
    std::vector<uint32_t> w_key, w_pay;  // lanes * ef entries, sorted by key per query
    bool has_w = false;                  // false before the first update: W holds sentinels
};

struct UpdateIn {
    uint32_t lanes = 1;
    bool open_head = true;   // open the popped address (single query) or keep it shared (batch)
    bool open_keys = false;  // last update only: also open the distances in W

    // All share vectors hold this party's shares.
    std::vector<std::vector<uint32_t>>* q_share = nullptr;    // lanes x D, arithmetic
    std::vector<std::vector<uint32_t>>* vec_share = nullptr;  // (K + seeds) x D, arithmetic
    std::vector<uint32_t>* flag_share = nullptr;              // K, XOR; may be null (all zero)
    std::vector<uint32_t>* a_share = nullptr;                 // K, graph-table address
    std::vector<uint32_t>* b_share = nullptr;                 // K + seeds, result address

    // Seeds are scored like candidates but enter only W, in place of its first
    // sentinels.  Used for the entry point of a layer, whose graph row is read
    // directly.  Only allowed in the first update of a layer.
    uint32_t seeds = 0;
    const std::vector<uint32_t>* w_sentinels = nullptr;       // ef public payloads
};

struct UpdateOut {
    std::vector<uint32_t> head;        // lanes values: opened, or this party's shares
    State state;
    std::vector<uint32_t> w_pay_open;  // last update: lanes * ef
    std::vector<uint32_t> w_key_open;  // last update with open_keys
    Cost cost;
};

namespace detail {

// [x lane 0 | y lane 0 | x lane 1 | y lane 1 | ...] from two lane-major arrays.
inline YArray ConcatLanes(BooleanCircuit* yc, const YArray& x, uint32_t nx,
                          const YArray& y, uint32_t ny, uint32_t lanes) {
    if (nx == 0) return y;
    if (ny == 0) return x;
    YArray both = Concat(yc, x, y);
    if (lanes == 1) return both;
    std::vector<uint32_t> pos;
    pos.reserve((size_t)lanes * (nx + ny));
    for (uint32_t j = 0; j < lanes; j++) {
        for (uint32_t i = 0; i < nx; i++) pos.push_back(j * nx + i);
        for (uint32_t i = 0; i < ny; i++) pos.push_back(lanes * nx + j * ny + i);
    }
    return Gather(yc, both, pos);
}

// Positions [lo, hi) of every lane of a lane-major array with `width` per lane.
inline std::vector<uint32_t> LaneRange(uint32_t lanes, uint32_t width, uint32_t lo, uint32_t hi) {
    std::vector<uint32_t> pos;
    pos.reserve((size_t)lanes * (hi - lo));
    for (uint32_t j = 0; j < lanes; j++) {
        for (uint32_t i = lo; i < hi; i++) pos.push_back(j * width + i);
    }
    return pos;
}

}  // namespace detail

inline UpdateOut RunUpdate(ABYParty* party, const Params& p, const Plan& plan,
                           const UpdateIn& in, State& st) {
    party->Reset();
    std::vector<Sharing*>& sharings = party->GetSharings();
    ArithmeticCircuit* ac = (ArithmeticCircuit*)sharings[S_ARITH]->GetCircuitBuildRoutine();
    BooleanCircuit* bc = (BooleanCircuit*)sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yc = (BooleanCircuit*)sharings[S_YAO]->GetCircuitBuildRoutine();
    ShareBin bin;

    const uint32_t L = in.lanes;
    const uint32_t K = plan.K;
    const uint32_t S = in.seeds;
    const uint32_t KS = K + S;
    const uint32_t kb = p.key_bits;
    const uint32_t ib = p.id_bits;
    assert(S == 0 || !st.has_w);
    assert(S <= p.ef);

    // --- squared L2 distances: one SIMD lane per (query, vector) pair --------
    std::vector<share*> terms;
    terms.reserve(p.D);
    std::vector<uint32_t> col((size_t)L * KS);
    for (uint32_t j = 0; j < p.D; j++) {
        for (uint32_t u = 0; u < L; u++) {
            for (uint32_t i = 0; i < KS; i++) col[(size_t)u * KS + i] = (*in.vec_share)[i][j];
        }
        share* v = bin.Keep(ac->PutSharedSIMDINGate(L * KS, col.data(), p.arith_bits));
        for (uint32_t u = 0; u < L; u++) {
            for (uint32_t i = 0; i < KS; i++) col[(size_t)u * KS + i] = (*in.q_share)[u][j];
        }
        share* q = bin.Keep(ac->PutSharedSIMDINGate(L * KS, col.data(), p.arith_bits));
        share* d = bin.Keep(ac->PutSUBGate(v, q));
        terms.push_back(bin.Keep(ac->PutMULGate(d, d)));
    }
    while (terms.size() > 1) {
        std::vector<share*> next;
        next.reserve((terms.size() + 1) / 2);
        for (size_t j = 0; j + 1 < terms.size(); j += 2) {
            next.push_back(bin.Keep(ac->PutADDGate(terms[j], terms[j + 1])));
        }
        if (terms.size() % 2 == 1) next.push_back(terms.back());
        terms.swap(next);
    }
    share* dist_y = bin.Keep(yc->PutA2YGate(terms[0]));

    YArray dist_all;
    dist_all.n = L * KS;
    for (uint32_t l = 0; l < kb; l++) dist_all.wires.push_back(dist_y->get_wire_id(l));
    YArray dist = (S == 0) ? dist_all : Gather(yc, dist_all, detail::LaneRange(L, KS, 0, K));

    // --- key = flag ? d_max : distance  (d_max is all ones) -----------------
    YArray cand;
    cand.n = L * K;
    if (in.flag_share != nullptr) {
        std::vector<uint32_t> fw = SharedToYaoWires(bc, yc, bin, *in.flag_share, 1);
        const uint32_t flag = Tile(yc, fw[0], L);
        for (uint32_t l = 0; l < kb; l++) {
            uint32_t s = yc->PutANDGate(flag, yc->PutINVGate(dist.wires[l]));
            cand.wires.push_back(yc->PutXORGate(dist.wires[l], s));
        }
    } else {
        cand.wires = dist.wires;
    }

    // --- payloads -----------------------------------------------------------
    std::vector<uint32_t> idx_ka = Range(0, kb);  // key | a
    std::vector<uint32_t> idx_kb = Range(0, kb);  // key | b
    if (!plan.last) {
        std::vector<uint32_t> aw = SharedToYaoWires(bc, yc, bin, *in.a_share, ib);
        for (uint32_t l = 0; l < ib; l++) {
            idx_ka.push_back((uint32_t)cand.wires.size());
            cand.wires.push_back(Tile(yc, aw[l], L));
        }
    }
    std::vector<uint32_t> seed_b;  // S values per wire
    {
        std::vector<uint32_t> bw = SharedToYaoWires(bc, yc, bin, *in.b_share, ib);
        std::vector<uint32_t> first = Range(0, K);
        std::vector<uint32_t> rest = Range(K, KS);
        for (uint32_t l = 0; l < ib; l++) {
            uint32_t w = (S == 0) ? bw[l] : yc->PutSubsetGate(bw[l], first.data(), K, true);
            idx_kb.push_back((uint32_t)cand.wires.size());
            cand.wires.push_back(Tile(yc, w, L));
            if (S > 0) seed_b.push_back(yc->PutSubsetGate(bw[l], rest.data(), S, true));
        }
    }

    const uint32_t ks = std::max(plan.kc, plan.kw);
    YArray sorted = ApplyNetwork(yc, cand, plan.sort_cand, kb, L);

    share* head_out = nullptr;
    share* c_key_out = nullptr;
    share* c_pay_out = nullptr;
    share* w_key_out = nullptr;
    share* w_pay_out = nullptr;

    // --- C: merge, pop the head, keep the rest shared -----------------------
    if (!plan.last) {
        YArray cand_c = Gather(yc, SelectWires(sorted, idx_ka), detail::LaneRange(L, ks, 0, plan.kc));
        YArray c_old;
        c_old.n = L * plan.c_in;
        if (plan.c_in > 0) {
            c_old.wires = SharedToYaoWires(bc, yc, bin, st.c_key, kb);
            std::vector<uint32_t> pw = SharedToYaoWires(bc, yc, bin, st.c_pay, ib);
            c_old.wires.insert(c_old.wires.end(), pw.begin(), pw.end());
        }
        YArray c_in = detail::ConcatLanes(yc, c_old, plan.c_in, cand_c, plan.kc, L);
        YArray c_new = ApplyNetwork(yc, c_in, plan.merge_c, kb, L);

        YArray head = Gather(yc, SelectWires(c_new, Range(kb, kb + ib)),
                             detail::LaneRange(L, plan.c_keep, 0, 1));
        head_out = in.open_head ? OpenYao(bc, yc, bin, head.wires)
                                : YaoToSharedOut(bc, yc, bin, head.wires);
        if (plan.c_keep > 1) {
            YArray tail = Gather(yc, c_new, detail::LaneRange(L, plan.c_keep, 1, plan.c_keep));
            c_key_out = YaoToSharedOut(bc, yc, bin,
                std::vector<uint32_t>(tail.wires.begin(), tail.wires.begin() + kb));
            c_pay_out = YaoToSharedOut(bc, yc, bin,
                std::vector<uint32_t>(tail.wires.begin() + kb, tail.wires.end()));
        }
    }

    // --- W: merge and truncate ----------------------------------------------
    {
        YArray cand_w = Gather(yc, SelectWires(sorted, idx_kb), detail::LaneRange(L, ks, 0, plan.kw));
        YArray w_old;
        w_old.n = L * p.ef;
        if (st.has_w) {
            w_old.wires = SharedToYaoWires(bc, yc, bin, st.w_key, kb);
            std::vector<uint32_t> pw = SharedToYaoWires(bc, yc, bin, st.w_pay, ib);
            w_old.wires.insert(w_old.wires.end(), pw.begin(), pw.end());
        } else {
            // First update: W = seeds (already in rank order for S <= 1) followed by sentinels.
            assert(S <= 1);
            std::vector<uint32_t> keys((size_t)L * (p.ef - S), p.dmax);
            std::vector<uint32_t> pays;
            pays.reserve(keys.size());
            for (uint32_t u = 0; u < L; u++) {
                for (uint32_t i = 0; i + S < p.ef; i++) pays.push_back((*in.w_sentinels)[i]);
            }
            YArray rest;
            rest.n = L * (p.ef - S);
            if (rest.n > 0) {
                rest = ConstArray(yc, keys, kb);
                YArray pay = ConstArray(yc, pays, ib);
                rest.wires.insert(rest.wires.end(), pay.wires.begin(), pay.wires.end());
            }
            if (S > 0) {
                YArray seed = Gather(yc, dist_all, detail::LaneRange(L, KS, K, KS));
                for (uint32_t l = 0; l < ib; l++) seed.wires.push_back(Tile(yc, seed_b[l], L));
                w_old = detail::ConcatLanes(yc, seed, S, rest, p.ef - S, L);
            } else {
                w_old = rest;
            }
        }
        YArray w_in = detail::ConcatLanes(yc, w_old, p.ef, cand_w, plan.kw, L);
        YArray w_new = ApplyNetwork(yc, w_in, plan.merge_w, kb, L);
        std::vector<uint32_t> kw(w_new.wires.begin(), w_new.wires.begin() + kb);
        std::vector<uint32_t> pw(w_new.wires.begin() + kb, w_new.wires.end());
        if (plan.last) {
            w_pay_out = OpenYao(bc, yc, bin, pw);
            if (in.open_keys) w_key_out = OpenYao(bc, yc, bin, kw);
        } else {
            w_key_out = YaoToSharedOut(bc, yc, bin, kw);
            w_pay_out = YaoToSharedOut(bc, yc, bin, pw);
        }
    }

    UpdateOut res;
    res.cost = Execute(party);

    if (!plan.last) {
        res.head = ReadSimd(head_out, L);
        if (plan.c_keep > 1) {
            res.state.c_key = ReadSimd(c_key_out, L * (plan.c_keep - 1));
            res.state.c_pay = ReadSimd(c_pay_out, L * (plan.c_keep - 1));
        }
        res.state.w_key = ReadSimd(w_key_out, L * p.ef);
        res.state.w_pay = ReadSimd(w_pay_out, L * p.ef);
        res.state.has_w = true;
    } else {
        res.w_pay_open = ReadSimd(w_pay_out, L * p.ef);
        if (in.open_keys) res.w_key_open = ReadSimd(w_key_out, L * p.ef);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Plaintext replay of an update, for one query
// ---------------------------------------------------------------------------

typedef std::pair<uint32_t, uint32_t> Entry;  // (key, payload)

struct PlainState {
    std::vector<Entry> c;  // after the pop
    std::vector<Entry> w;
    bool has_w = false;
};

// cand: the K scored candidates of this query, (key, address).  seeds: scored
// seeds, (key, address).  Returns the popped address (undefined if plan.last).
inline uint32_t PlainUpdate(const Params& p, const Plan& plan, const std::vector<Entry>& cand,
                            const std::vector<Entry>& seeds,
                            const std::vector<uint32_t>& w_sentinels, PlainState& st) {
    std::vector<Entry> sorted = RunNetworkPlain(plan.sort_cand, cand);
    if (!st.has_w) {
        st.w.assign(seeds.begin(), seeds.end());
        for (uint32_t i = 0; i + seeds.size() < p.ef; i++) st.w.emplace_back(p.dmax, w_sentinels[i]);
        st.has_w = true;
    }
    std::vector<Entry> w_in(st.w);
    w_in.insert(w_in.end(), sorted.begin(), sorted.begin() + plan.kw);
    st.w = RunNetworkPlain(plan.merge_w, w_in);
    if (plan.last) return 0;
    std::vector<Entry> c_in(st.c);
    c_in.insert(c_in.end(), sorted.begin(), sorted.begin() + plan.kc);
    std::vector<Entry> c_new = RunNetworkPlain(plan.merge_c, c_in);
    st.c.assign(c_new.begin() + 1, c_new.end());
    return c_new[0].second;
}

// ---------------------------------------------------------------------------
// One query on one layer (Algorithm 2)
// ---------------------------------------------------------------------------

struct LayerSearchResult {
    bool ok = true;
    std::vector<uint32_t> w_open;  // opened result addresses, best first
    std::vector<Entry> w_ref;      // plaintext reference of W
    Cost access;
    Cost update;
};

// Runs T iterations for query number `qi` from entry point `entry` and checks
// every opened value against a plaintext replay.  Dummy addresses are taken
// from [dum_base, dum_base + T*M) and the W sentinels from
// [w_sent_base, w_sent_base + ef).
inline LayerSearchResult SearchLayer(ABYParty* party, e_role role, const Params& p,
                                     const Layer& layer, uint32_t qi, uint32_t entry,
                                     uint32_t dum_base, uint32_t w_sent_base,
                                     ShareStream& shares, uint32_t verbose) {
    const uint32_t M = p.M;
    const uint32_t D = p.D;
    const uint32_t T = p.T;
    const uint32_t ef = p.ef;
    const uint32_t id_bitlen = p.id_bits;
    (void)role;
    LayerSearchResult res;
    const std::vector<uint32_t> query = layer.Query(qi);

    std::vector<std::vector<uint32_t>> q_share(1, std::vector<uint32_t>(D));
    for (uint32_t j = 0; j < D; j++) q_share[0][j] = shares.Add(query[j], p.arith_bits);

    std::vector<uint32_t> w_sentinels(ef);
    for (uint32_t j = 0; j < ef; j++) w_sentinels[j] = w_sent_base + j;

    auto plain_dist = [&](const std::vector<uint32_t>& v) -> uint32_t {
        uint32_t s = 0;
        for (uint32_t j = 0; j < D; j++) {
            uint32_t d = v[j] - query[j];
            s += d * d;
        }
        return s & p.dmax;
    };

    // Addresses opened so far.  The entry point and sigma are there from the start.
    std::vector<uint32_t> visited;
    visited.reserve(2 + (size_t)T * M);
    visited.push_back(entry);
    visited.push_back(layer.sigma);
    std::unordered_set<uint32_t> visited_set(visited.begin(), visited.end());

    // Plaintext reference of the secret state.  Both processes hold the
    // plaintext tables in this harness, so each can replay the iteration and
    // check every opened value against it.
    State state;
    PlainState ref;
    uint32_t u = entry;
    bool ok = true;

    for (uint32_t t = 0; t < T && ok; t++) {
        const Plan plan = MakePlan(p, M, T - 1 - t, (uint32_t)state.c_key.size());

        // ---- visited check -------------------------------------------------
        std::vector<uint32_t> neigh = layer.Neighbors(u);
        std::vector<uint32_t> neigh_share(M), dummy_share(M), dummy(M);
        for (uint32_t i = 0; i < M; i++) {
            dummy[i] = dum_base + t * M + i;
            neigh_share[i] = shares.Xor(neigh[i], id_bitlen);
            dummy_share[i] = shares.Xor(dummy[i], id_bitlen);
        }
        AccessOut va = RunAccessCheck(party, id_bitlen, neigh_share, dummy_share, visited, Dedup::kNone);
        res.access.Add(va.cost);

        std::vector<uint32_t> opened_ref;
        std::vector<uint8_t> flag_ref;
        PlainAccessCheck(neigh, dummy, visited_set, Dedup::kNone, opened_ref, flag_ref);
        for (uint32_t i = 0; i < M && ok; i++) {
            if (va.opened[i] != opened_ref[i]) {
                std::cerr << "[t=" << t << "] visited check mismatch at " << i << ": opened "
                          << va.opened[i] << ", expected " << opened_ref[i] << std::endl;
                ok = false;
            } else if (!visited_set.insert(va.opened[i]).second) {
                std::cerr << "[t=" << t << "] address " << va.opened[i]
                          << " opened twice (repeated neighbour in the graph row of " << u << ")" << std::endl;
                ok = false;
            }
            visited.push_back(va.opened[i]);
        }
        if (!ok) break;

        // ---- fetch D[u'] and update ----------------------------------------
        // Iteration 0 also scores the entry point, which goes straight into W.
        const uint32_t seeds = (t == 0) ? 1U : 0U;
        std::vector<std::vector<uint32_t>> vec(M + seeds), vec_share(M + seeds, std::vector<uint32_t>(D));
        std::vector<uint32_t> a_share(M), b_share(M + seeds);
        for (uint32_t i = 0; i < M + seeds; i++) {
            const uint32_t id = (i < M) ? va.opened[i] : entry;
            vec[i] = layer.Vector(id);
            for (uint32_t j = 0; j < D; j++) vec_share[i][j] = shares.Add(vec[i][j], p.arith_bits);
            if (i < M) a_share[i] = shares.Xor(id, id_bitlen);
            b_share[i] = shares.Xor(id, id_bitlen);
        }

        UpdateIn in;
        in.lanes = 1;
        in.open_head = true;
        in.open_keys = (verbose != 0);
        in.q_share = &q_share;
        in.vec_share = &vec_share;
        in.flag_share = &va.flag_share;
        in.a_share = &a_share;
        in.b_share = &b_share;
        in.seeds = seeds;
        in.w_sentinels = &w_sentinels;
        UpdateOut ub = RunUpdate(party, p, plan, in, state);
        res.update.Add(ub.cost);

        // ---- reference replay ----------------------------------------------
        std::vector<Entry> cand(M), seed_ref;
        for (uint32_t i = 0; i < M; i++) {
            cand[i] = Entry(flag_ref[i] ? p.dmax : plain_dist(vec[i]), va.opened[i]);
        }
        if (seeds) seed_ref.emplace_back(plain_dist(vec[M]), entry);
        const uint32_t next_ref = PlainUpdate(p, plan, cand, seed_ref, w_sentinels, ref);
        if (!plan.last) {
            if (ub.head[0] != next_ref) {
                std::cerr << "[t=" << t << "] head of C mismatch: opened " << ub.head[0]
                          << ", expected " << next_ref << std::endl;
                ok = false;
                break;
            }
            state = ub.state;
            u = ub.head[0];
        } else {
            res.w_open = ub.w_pay_open;
            for (uint32_t j = 0; j < ef; j++) {
                if (res.w_open[j] != ref.w[j].second ||
                    (in.open_keys && ub.w_key_open[j] != ref.w[j].first)) {
                    std::cerr << "W mismatch at rank " << j << ": opened " << res.w_open[j]
                              << ", expected " << ref.w[j].second << std::endl;
                    ok = false;
                }
            }
        }

        if (verbose) {
            std::cout << "[t=" << t << "] visited: " << va.cost.online_ms << " ms, "
                      << va.cost.online_bytes << " B, " << va.cost.rounds << " rounds, |V|="
                      << (visited.size() - M)
                      << " | update: " << ub.cost.online_ms << " ms, "
                      << ub.cost.online_bytes << " B, " << ub.cost.rounds << " rounds" << std::endl;
        }
    }
    res.ok = ok;
    res.w_ref = ref.w;
    return res;
}

}  // namespace hnsecw

#endif  // HNSECW_SEARCH_H
