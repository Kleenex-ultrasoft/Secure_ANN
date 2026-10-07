// HNSecW multiple queries on one shuffled copy of the tables, two-party (ABY).
//
// The queries are searched one after another.  The lists of opened addresses
// (one for the vector table, one for the graph table) are shared by all
// queries, so no table address is ever opened twice and the tables do not
// have to be reshuffled between queries.
//
// A later query cannot fetch a vector that an earlier query fetched.  It does
// not have to: the addresses opened so far are public, so the shares behind
// them can be read again.  Every query therefore starts by scoring all
// vectors fetched so far and keeps the best of them as its initial C and W.
// From then on
//
//   * a neighbour that is in the visited list has already been offered to
//     this query and is replaced by a dummy, exactly as in a single query;
//   * a popped node whose graph row was opened before has nothing new to
//     offer (its neighbours are all in the visited list) and reads a dummy
//     row.
//
// Per query: the entry point goes through the visited check, the catch-up
// update scores the fetched vectors, and T iterations follow, each with a
// graph access, a vector access and an update, as in the batch program with
// one query (hnsecw_batch_b2y.cpp).

#include "hnsecw_multi_b2y.h"
#include "hnsecw_search.h"

using namespace hnsecw;

int32_t hnsecw_multi_b2y(
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

    // Dummy pool:
    //   U        replacements for entry points that were fetched before
    //   U*T*M    replacements for visited neighbours
    //   U*T      replacements for repeated graph accesses
    //   ef       sentinels that initialise W
    Layer layer;
    if (!InitLayer(layer, M, D, bitlen, id_bitlen,
                   (uint64_t)U + (uint64_t)U * T * M + (uint64_t)U * T + ef)) return 1;
    const uint32_t N = layer.N;
    const uint32_t entry_dum_base = N;                           // + u
    const uint32_t neigh_dum_base = N + U;                       // + (u*T + t)*M + i
    const uint32_t graph_dum_base = neigh_dum_base + U * T * M;  // + u*T + t
    const uint32_t w_sent_base = graph_dum_base + U * T;         // + j

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
    std::cout << "  HNSecW multiple queries on one shuffled copy, one layer (2PC)" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "  queries=" << U << " N=" << N << (layer.real ? " (data file)" : " (synthetic)")
              << " M=" << M << " D=" << D << " T=" << T << " ef=" << ef << std::endl;
    std::cout << "  coordinate bits=" << bitlen << " address bits=" << id_bitlen
              << " distance bits=" << p.key_bits << std::endl;
    std::cout << "========================================================\n" << std::endl;

    ShareStream shares(role, 0x484e53656357ULL);
    std::vector<uint32_t> w_sentinels(ef);
    for (uint32_t j = 0; j < ef; j++) w_sentinels[j] = w_sent_base + j;

    // Opened addresses of the vector table and of the graph table, shared by
    // all queries.
    std::vector<uint32_t> visited(1, layer.sigma);
    std::unordered_set<uint32_t> visited_set(visited.begin(), visited.end());
    std::vector<uint32_t> expanded;
    std::unordered_set<uint32_t> expanded_set;

    ABYParty* party = new ABYParty(role, address, port, seclvl, p.arith_bits, nthreads, mt_alg, 4000000);

    Cost cost_graph, cost_vector, cost_update, cost_catchup;
    std::vector<uint32_t> best(U), topk((size_t)U * ef, 0xFFFFFFFFu);
    bool ok = true;

    // Runs an access check and compares the opened addresses with the replay.
    auto access = [&](std::vector<uint32_t>& addr_share, const std::vector<uint32_t>& addr_ref,
                      const std::vector<uint32_t>& dummy,
                      std::vector<uint32_t>& opened_list, std::unordered_set<uint32_t>& opened_set,
                      Cost& cost, const std::string& label, AccessOut& out) -> bool {
        const uint32_t n = (uint32_t)dummy.size();
        std::vector<uint32_t> dummy_share(n);
        for (uint32_t i = 0; i < n; i++) dummy_share[i] = shares.Xor(dummy[i], id_bitlen);
        // The addresses of one check are distinct: one entry point, one popped
        // address, or the neighbours of one node.
        out = RunAccessCheck(party, id_bitlen, addr_share, dummy_share, opened_list, Dedup::kNone);
        cost.Add(out.cost);
        if (verbose) PrintCost(label, out.cost);
        std::vector<uint32_t> opened_ref;
        std::vector<uint8_t> flag_ref;
        PlainAccessCheck(addr_ref, dummy, opened_set, Dedup::kNone, opened_ref, flag_ref);
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

    for (uint32_t u = 0; u < U && ok; u++) {
        const std::vector<uint32_t> query = layer.Query(u);
        std::vector<std::vector<uint32_t>> q_share(1, std::vector<uint32_t>(D));
        for (uint32_t j = 0; j < D; j++) q_share[0][j] = shares.Add(query[j], p.arith_bits);

        State state;
        PlainState ref;
        std::vector<uint32_t> head_share, head_ref(1, 0);
        std::vector<uint32_t> w_open;

        // Scores the fetched vectors `ids` for this query, runs the update
        // circuit and replays it in plaintext.  An address beyond the real
        // rows is a dummy; its flag stands for the dummy mark stored in the
        // table row.
        auto update = [&](const Plan& plan, const std::vector<uint32_t>& ids, Cost& cost,
                          const std::string& label) -> bool {
            const uint32_t n = (uint32_t)ids.size();
            std::vector<std::vector<uint32_t>> vec(n), vec_share(n, std::vector<uint32_t>(D));
            std::vector<uint32_t> a_share(n), b_share(n), flag_share(n);
            std::vector<Entry> cand(n);
            for (uint32_t i = 0; i < n; i++) {
                vec[i] = layer.Vector(ids[i]);
                for (uint32_t j = 0; j < D; j++) vec_share[i][j] = shares.Add(vec[i][j], p.arith_bits);
                a_share[i] = shares.Xor(ids[i], id_bitlen);
                b_share[i] = shares.Xor(ids[i], id_bitlen);
                const bool dummy = ids[i] >= N;
                flag_share[i] = shares.Xor(dummy ? 1 : 0, 1);
                uint32_t s = 0;
                for (uint32_t j = 0; j < D; j++) {
                    uint32_t d = vec[i][j] - query[j];
                    s += d * d;
                }
                cand[i] = Entry(dummy ? p.dmax : (s & p.dmax), ids[i]);
            }
            UpdateIn in;
            in.lanes = 1;
            in.open_head = false;
            in.open_keys = (verbose != 0);
            in.q_share = &q_share;
            in.vec_share = &vec_share;
            in.flag_share = &flag_share;
            in.a_share = &a_share;
            in.b_share = &b_share;
            in.w_sentinels = &w_sentinels;
            UpdateOut ub = RunUpdate(party, p, plan, in, state);
            cost.Add(ub.cost);
            if (verbose) PrintCost(label, ub.cost);

            head_ref[0] = PlainUpdate(p, plan, cand, std::vector<Entry>(), w_sentinels, ref);
            if (!plan.last) {
                state = ub.state;
                head_share = ub.head;
                return true;
            }
            w_open = ub.w_pay_open;
            for (uint32_t j = 0; j < ef; j++) {
                if (w_open[j] != ref.w[j].second || (in.open_keys && ub.w_key_open[j] != ref.w[j].first)) {
                    std::cerr << "W mismatch for query " << u << " at rank " << j << ": opened "
                              << w_open[j] << ", expected " << ref.w[j].second << std::endl;
                    return false;
                }
            }
            return true;
        };

        const std::string qtag = "[query " + std::to_string(u) + "]";

        // ---- entry point ---------------------------------------------------
        {
            std::vector<uint32_t> addr_share(1, shares.Xor(entry[u], id_bitlen));
            std::vector<uint32_t> addr_ref(1, entry[u]), dummy(1, entry_dum_base + u);
            AccessOut ao;
            ok = access(addr_share, addr_ref, dummy, visited, visited_set, cost_vector,
                        qtag + " entry access:", ao);
            if (!ok) break;
        }

        // ---- catch-up: everything fetched so far ---------------------------
        {
            std::vector<uint32_t> ids(visited.begin() + 1, visited.end());  // without sigma
            Plan plan = MakePlan(p, (uint32_t)ids.size(), T, 0);
            ok = update(plan, ids, cost_catchup, qtag + " catch-up (" + std::to_string(ids.size()) + " vectors):");
            if (!ok) break;
        }

        for (uint32_t t = 0; t < T && ok; t++) {
            const std::string tag = qtag + "[t=" + std::to_string(t) + "]";

            // ---- graph access ----------------------------------------------
            std::vector<uint32_t> dummy_g(1, graph_dum_base + u * T + t);
            AccessOut ga;
            ok = access(head_share, head_ref, dummy_g, expanded, expanded_set, cost_graph,
                        tag + " graph access:", ga);
            if (!ok) break;

            // ---- vector access ---------------------------------------------
            std::vector<uint32_t> neigh = layer.Neighbors(ga.opened[0]);
            std::vector<uint32_t> neigh_share(M), dummy_v(M);
            for (uint32_t i = 0; i < M; i++) {
                neigh_share[i] = shares.Xor(neigh[i], id_bitlen);
                dummy_v[i] = neigh_dum_base + (u * T + t) * M + i;
            }
            AccessOut va;
            ok = access(neigh_share, neigh, dummy_v, visited, visited_set, cost_vector,
                        tag + " vector access:", va);
            if (!ok) break;

            // ---- update ----------------------------------------------------
            Plan plan = MakePlan(p, M, T - 1 - t, (uint32_t)state.c_key.size());
            ok = update(plan, va.opened, cost_update, tag + " update:");
        }
        if (!ok) break;

        best[u] = entry[u];
        bool found = false;
        for (uint32_t j = 0; j < ef; j++) {
            uint32_t id = w_open[j];
            if (id >= N) continue;
            topk[(size_t)u * ef + j] = id;
            if (!found) {
                best[u] = id;
                found = true;
            }
        }
    }
    delete party;

    if (!ok) {
        std::cerr << "[FAILED] the opened values disagree with the plaintext reference" << std::endl;
        return 2;
    }

    if (role == CLIENT && !entry_out.empty()) {
        // entry_out: best real node per query.  entry_out.topk: U rows of ef
        // ids, 0xFFFFFFFF where W holds a dummy.
        WriteIdsToFile(entry_out, best);
        WriteIdsToFile(entry_out + ".topk", topk);
    }

    std::cout << "[Verified] every opened value matches the plaintext reference" << std::endl;
    PrintCost("[Graph access]", cost_graph);
    PrintCost("[Vector access]", cost_vector);
    PrintCost("[Catch-up]", cost_catchup);
    PrintCost("[Distance+update]", cost_update);
    Cost total = cost_graph;
    total.Add(cost_vector);
    total.Add(cost_catchup);
    total.Add(cost_update);
    PrintTotals(total);
    return 0;
}
