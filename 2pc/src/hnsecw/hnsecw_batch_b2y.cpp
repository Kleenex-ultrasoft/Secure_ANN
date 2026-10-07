// HNSecW batch-query layer search, two-party (ABY).
//
// One call runs Algorithm 4 (ParallelBatchSearchLayer) on one layer for U
// queries at once.  All queries share one visited list and every fetched
// vector is scored by every query, so a vector is read from the table at most
// once for the whole batch.
//
// First the U entry points are deduplicated, fetched and scored.  Then, for
// T = L_C iterations:
//
//   1. Graph access.  The U popped addresses (one per query, secret) are
//      checked against the graph rows opened so far and against each other;
//      repeats are replaced by dummy addresses and the U addresses are
//      opened.
//   2. Vector access.  The U*M neighbour addresses are checked against the
//      visited list and deduplicated by an oblivious sort; repeats are
//      replaced by dummies and the U*M addresses are opened.
//   3. Update.  Every query scores all U*M fetched vectors and merges them
//      into its own C and W.  The heads of the new C arrays stay shared and
//      feed step 1 of the next iteration.
//
// The W arrays are opened after the last iteration.

#include "hnsecw_batch_b2y.h"
#include "hnsecw_search.h"

using namespace hnsecw;

namespace {

Dedup PickDedup(uint32_t n) {
    // Comparing all pairs in GMW costs n^2/2 equality tests but few rounds;
    // sorting in Yao costs n log^2 n comparators.
    return (n <= 16) ? Dedup::kPairwise : Dedup::kSort;
}

}  // namespace

int32_t hnsecw_batch_b2y(
    e_role role, const std::string& address, uint16_t port, seclvl seclvl,
    uint32_t num_queries, uint32_t M, uint32_t D, uint32_t L_C, uint32_t L_W,
    uint32_t bitlen, uint32_t id_bitlen,
    uint32_t entry_id, const std::string& entry_file, const std::string& entry_out,
    uint32_t verbose, uint32_t nthreads, e_mt_gen_alg mt_alg) {

    const uint32_t U = num_queries;
    if (U == 0) {
        std::cerr << "num_queries must be positive" << std::endl;
        return 1;
    }
    Params p;
    if (!InitParams(p, M, D, L_C, L_W, bitlen, id_bitlen)) return 1;
    const uint32_t T = p.T;
    const uint32_t ef = p.ef;
    const uint32_t K = U * M;  // vectors fetched per iteration

    // Dummy pool:
    //   U        replacements for repeated entry points
    //   T*K      replacements for visited or repeated neighbours
    //   T*U      replacements for repeated graph accesses
    //   ef       sentinels that initialise W
    Layer layer;
    if (!InitLayer(layer, M, D, bitlen, id_bitlen, (uint64_t)U + (uint64_t)T * K + (uint64_t)T * U + ef)) return 1;
    const uint32_t N = layer.N;
    const uint32_t entry_dum_base = N;                   // + u
    const uint32_t neigh_dum_base = N + U;               // + t*K + i
    const uint32_t graph_dum_base = neigh_dum_base + T * K;  // + t*U + u
    const uint32_t w_sent_base = graph_dum_base + T * U;     // + j

    std::vector<std::vector<uint32_t>> query(U);
    for (uint32_t u = 0; u < U; u++) query[u] = layer.Query(u);

    std::vector<uint32_t> entry(U, 12345u % N);
    if (!entry_file.empty()) {
        std::vector<uint32_t> ids = ReadIdsFromFile(entry_file);
        if (ids.size() < U) {
            std::cerr << "entry file holds " << ids.size() << " ids, need " << U << std::endl;
            return 1;
        }
        for (uint32_t u = 0; u < U; u++) entry[u] = ids[u];
    } else if (entry_id != kEntryAuto) {
        entry.assign(U, entry_id);
    }
    for (uint32_t u = 0; u < U; u++) {
        if (entry[u] >= N) {
            std::cerr << "entry point " << entry[u] << " is not a node of this layer (N=" << N << ")" << std::endl;
            return 1;
        }
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "  HNSecW batch query, one layer (2PC)" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "  queries=" << U << " N=" << N << (layer.real ? " (data file)" : " (synthetic)")
              << " M=" << M << " D=" << D << " T=" << T << " ef=" << ef << std::endl;
    std::cout << "  coordinate bits=" << bitlen << " address bits=" << id_bitlen
              << " distance bits=" << p.key_bits << std::endl;
    std::cout << "========================================================\n" << std::endl;

    ShareStream shares(role, 0x484e53656357ULL);

    std::vector<std::vector<uint32_t>> q_share(U, std::vector<uint32_t>(D));
    for (uint32_t u = 0; u < U; u++) {
        for (uint32_t j = 0; j < D; j++) q_share[u][j] = shares.Add(query[u][j], p.arith_bits);
    }
    std::vector<uint32_t> w_sentinels(ef);
    for (uint32_t j = 0; j < ef; j++) w_sentinels[j] = w_sent_base + j;

    auto plain_dist = [&](uint32_t u, const std::vector<uint32_t>& v) -> uint32_t {
        uint32_t s = 0;
        for (uint32_t j = 0; j < D; j++) {
            uint32_t d = v[j] - query[u][j];
            s += d * d;
        }
        return s & p.dmax;
    };

    // Opened addresses of the vector table (shared by all queries) and of the
    // graph table.
    std::vector<uint32_t> visited(1, layer.sigma);
    std::unordered_set<uint32_t> visited_set(visited.begin(), visited.end());
    std::vector<uint32_t> expanded;
    std::unordered_set<uint32_t> expanded_set;

    ABYParty* party = new ABYParty(role, address, port, seclvl, p.arith_bits, nthreads, mt_alg, 4000000);

    State state;
    std::vector<PlainState> ref(U);
    Cost cost_graph, cost_vector, cost_update;
    bool ok = true;
    std::vector<uint32_t> w_open;

    // Scores the fetched vectors `ids` (with their rejection flags) for every
    // query, runs the update circuit and replays it in plaintext.  On return
    // head_share / head_ref hold the popped addresses unless plan.last.
    std::vector<uint32_t> head_share, head_ref(U, 0);
    auto update = [&](const Plan& plan, const std::vector<uint32_t>& ids,
                      std::vector<uint32_t>& flag_share, const std::vector<uint8_t>& flag_ref,
                      const std::string& label) -> bool {
        const uint32_t n = (uint32_t)ids.size();
        std::vector<std::vector<uint32_t>> vec(n), vec_share(n, std::vector<uint32_t>(D));
        std::vector<uint32_t> a_share(n), b_share(n);
        for (uint32_t i = 0; i < n; i++) {
            vec[i] = layer.Vector(ids[i]);
            for (uint32_t j = 0; j < D; j++) vec_share[i][j] = shares.Add(vec[i][j], p.arith_bits);
            a_share[i] = shares.Xor(ids[i], id_bitlen);
            b_share[i] = shares.Xor(ids[i], id_bitlen);
        }
        UpdateIn in;
        in.lanes = U;
        in.open_head = false;
        in.open_keys = (verbose != 0);
        in.q_share = &q_share;
        in.vec_share = &vec_share;
        in.flag_share = &flag_share;
        in.a_share = &a_share;
        in.b_share = &b_share;
        in.w_sentinels = &w_sentinels;
        UpdateOut ub = RunUpdate(party, p, plan, in, state);
        cost_update.Add(ub.cost);
        if (verbose) PrintCost(label + " update:", ub.cost);

        for (uint32_t u = 0; u < U; u++) {
            std::vector<Entry> cand(n);
            for (uint32_t i = 0; i < n; i++) {
                cand[i] = Entry(flag_ref[i] ? p.dmax : plain_dist(u, vec[i]), ids[i]);
            }
            head_ref[u] = PlainUpdate(p, plan, cand, std::vector<Entry>(), w_sentinels, ref[u]);
        }
        if (!plan.last) {
            state = ub.state;
            head_share = ub.head;
            return true;
        }
        w_open = ub.w_pay_open;
        for (uint32_t u = 0; u < U; u++) {
            for (uint32_t j = 0; j < ef; j++) {
                if (w_open[u * ef + j] != ref[u].w[j].second ||
                    (in.open_keys && ub.w_key_open[u * ef + j] != ref[u].w[j].first)) {
                    std::cerr << "W mismatch for query " << u << " at rank " << j << ": opened "
                              << w_open[u * ef + j] << ", expected " << ref[u].w[j].second << std::endl;
                    return false;
                }
            }
        }
        return true;
    };

    // Runs an access check and compares the opened addresses with the replay.
    auto access = [&](std::vector<uint32_t>& addr_share, const std::vector<uint32_t>& addr_ref,
                      const std::vector<uint32_t>& dummy,
                      std::vector<uint32_t>& opened_list, std::unordered_set<uint32_t>& opened_set,
                      Cost& cost, const std::string& label,
                      AccessOut& out, std::vector<uint8_t>& flag_ref) -> bool {
        const uint32_t n = (uint32_t)dummy.size();
        std::vector<uint32_t> dummy_share(n);
        for (uint32_t i = 0; i < n; i++) dummy_share[i] = shares.Xor(dummy[i], id_bitlen);
        const Dedup dedup = PickDedup(n);
        out = RunAccessCheck(party, id_bitlen, addr_share, dummy_share, opened_list, dedup);
        cost.Add(out.cost);
        if (verbose) PrintCost(label, out.cost);
        std::vector<uint32_t> opened_ref;
        PlainAccessCheck(addr_ref, dummy, opened_set, dedup, opened_ref, flag_ref);
        for (uint32_t i = 0; i < n; i++) {
            if (out.opened[i] != opened_ref[i]) {
                std::cerr << label << " mismatch at " << i << ": opened " << out.opened[i]
                          << ", expected " << opened_ref[i] << std::endl;
                return false;
            }
            if (!opened_set.insert(out.opened[i]).second) {
                std::cerr << label << " address " << out.opened[i] << " opened twice" << std::endl;
                return false;
            }
            opened_list.push_back(out.opened[i]);
        }
        return true;
    };

    // ---- entry points ------------------------------------------------------
    {
        std::vector<uint32_t> addr_share(U), dummy(U);
        for (uint32_t u = 0; u < U; u++) {
            addr_share[u] = shares.Xor(entry[u], id_bitlen);
            dummy[u] = entry_dum_base + u;
        }
        AccessOut ao;
        std::vector<uint8_t> flag_ref;
        ok = access(addr_share, entry, dummy, visited, visited_set, cost_vector, "[entry] vector access:", ao, flag_ref);
        if (ok) {
            Plan plan = MakePlan(p, U, T, 0);
            ok = update(plan, ao.opened, ao.flag_share, flag_ref, "[entry]");
        }
    }

    for (uint32_t t = 0; t < T && ok; t++) {
        const std::string tag = "[t=" + std::to_string(t) + "]";

        // ---- graph access --------------------------------------------------
        std::vector<uint32_t> dummy_g(U);
        for (uint32_t u = 0; u < U; u++) dummy_g[u] = graph_dum_base + t * U + u;
        AccessOut ga;
        std::vector<uint8_t> gflag_ref;
        ok = access(head_share, head_ref, dummy_g, expanded, expanded_set, cost_graph,
                    tag + " graph access:", ga, gflag_ref);
        if (!ok) break;

        // ---- vector access -------------------------------------------------
        std::vector<uint32_t> neigh(K), neigh_share(K), dummy_v(K);
        for (uint32_t u = 0; u < U; u++) {
            std::vector<uint32_t> row = layer.Neighbors(ga.opened[u]);
            for (uint32_t i = 0; i < M; i++) {
                neigh[u * M + i] = row[i];
                neigh_share[u * M + i] = shares.Xor(row[i], id_bitlen);
                dummy_v[u * M + i] = neigh_dum_base + t * K + u * M + i;
            }
        }
        AccessOut va;
        std::vector<uint8_t> vflag_ref;
        ok = access(neigh_share, neigh, dummy_v, visited, visited_set, cost_vector,
                    tag + " vector access:", va, vflag_ref);
        if (!ok) break;

        // ---- update --------------------------------------------------------
        Plan plan = MakePlan(p, K, T - 1 - t, (uint32_t)(state.c_key.size() / U));
        ok = update(plan, va.opened, va.flag_share, vflag_ref, tag);
    }

    delete party;

    if (!ok) {
        std::cerr << "[FAILED] the opened values disagree with the plaintext reference" << std::endl;
        return 2;
    }

    if (role == CLIENT && !entry_out.empty()) {
        // entry_out: best real node per query.  entry_out.topk: U rows of ef
        // ids, 0xFFFFFFFF where W holds a dummy.
        std::vector<uint32_t> best(U), topk((size_t)U * ef, 0xFFFFFFFFu);
        for (uint32_t u = 0; u < U; u++) {
            best[u] = entry[u];
            bool found = false;
            for (uint32_t j = 0; j < ef; j++) {
                uint32_t id = w_open[u * ef + j];
                if (id >= N) continue;
                topk[(size_t)u * ef + j] = id;
                if (!found) {
                    best[u] = id;
                    found = true;
                }
            }
        }
        WriteIdsToFile(entry_out, best);
        WriteIdsToFile(entry_out + ".topk", topk);
    }

    std::cout << "[Verified] every opened value matches the plaintext reference" << std::endl;
    PrintCost("[Graph access]", cost_graph);
    PrintCost("[Vector access]", cost_vector);
    PrintCost("[Distance+update]", cost_update);
    Cost total = cost_graph;
    total.Add(cost_vector);
    total.Add(cost_update);
    PrintTotals(total);
    return 0;
}
