// HNSecW single-query layer search, two-party (ABY).
//
// One call runs Algorithm 2 (ObliviousSearchLayer) on one layer:
//   T = L_C iterations, each expanding the M neighbours of the node popped
//   from the candidate array C.
//
// Per iteration the servers run two circuits (hnsecw_search.h).
//
//   AccessCheck (GMW).  The neighbour addresses are XOR-shared.  Each one is
//   tested against the public list of addresses opened so far and, if it was
//   seen, replaced by a fresh dummy address.  The selected addresses are
//   opened; the hit bits stay shared.
//
//   Update (arithmetic + Yao).  Squared L2 distances of the M fetched vectors
//   are computed on arithmetic shares, converted to Yao, masked to d_max for
//   visited neighbours, sorted, and merged into C and W.  Only the address at
//   the head of the new C is opened.  C and W stay secret-shared between
//   iterations.
//
// The result array W is opened once, after the last iteration.

#include "hnsecw_single_b2y.h"
#include "hnsecw_search.h"

using namespace hnsecw;

int32_t hnsecw_single_b2y(
    e_role role, const std::string& address, uint16_t port, seclvl seclvl,
    uint32_t M, uint32_t D, uint32_t L_C, uint32_t L_W,
    uint32_t bitlen, uint32_t id_bitlen,
    uint32_t entry_id, const std::string& entry_file, const std::string& entry_out,
    uint32_t verbose, uint32_t nthreads, e_mt_gen_alg mt_alg) {

    Params p;
    if (!InitParams(p, M, D, L_C, L_W, bitlen, id_bitlen)) return 1;
    const uint32_t T = p.T;
    const uint32_t ef = p.ef;

    // Dummy pool: T*M replacements for visited neighbours, then ef sentinels
    // that initialise W.
    Layer layer;
    if (!InitLayer(layer, M, D, bitlen, id_bitlen, (uint64_t)T * M + ef)) return 1;
    const uint32_t N = layer.N;

    uint32_t qi = 0;
    if (const char* env = std::getenv("HNSECW_QUERY_INDEX")) qi = (uint32_t)std::strtoul(env, nullptr, 10);

    uint32_t entry = 12345u % N;
    if (!entry_file.empty()) {
        entry = ReadIdsFromFile(entry_file).at(0);
    } else if (entry_id != kEntryAuto) {
        entry = entry_id;
    }
    if (entry >= N) {
        std::cerr << "entry point " << entry << " is not a node of this layer (N=" << N << ")" << std::endl;
        return 1;
    }

    const Plan plan0 = MakePlan(p, M, T - 1, 0);
    std::cout << "\n========================================================" << std::endl;
    std::cout << "  HNSecW single query, one layer (2PC)" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "  N=" << N << (layer.real ? " (data file)" : " (synthetic)")
              << " M=" << M << " D=" << D << " T=" << T << " ef=" << ef << std::endl;
    std::cout << "  coordinate bits=" << bitlen << " address bits=" << id_bitlen
              << " distance bits=" << p.key_bits << std::endl;
    std::cout << "  comparators in iteration 0: sort=" << plan0.sort_cand.cmps.size()
              << " merge C=" << plan0.merge_c.cmps.size()
              << " merge W=" << plan0.merge_w.cmps.size() << std::endl;
    std::cout << "========================================================\n" << std::endl;

    // Both processes derive the same mask stream, so the shares they feed in
    // are consistent.  See hnsecw_core.h.
    ShareStream shares(role, 0x484e53656357ULL);

    ABYParty* party = new ABYParty(role, address, port, seclvl, p.arith_bits, nthreads, mt_alg, 4000000);
    LayerSearchResult res = SearchLayer(party, role, p, layer, qi, entry, N, N + T * M, shares, verbose);
    delete party;

    if (!res.ok) {
        std::cerr << "[FAILED] the opened values disagree with the plaintext reference" << std::endl;
        return 2;
    }

    if (role == CLIENT && !entry_out.empty()) {
        std::vector<uint32_t> real_ids;
        for (uint32_t id : res.w_open) {
            if (id < N) real_ids.push_back(id);
        }
        WriteIdsToFile(entry_out, std::vector<uint32_t>(1, real_ids.empty() ? entry : real_ids[0]));
        WriteIdsToFile(entry_out + ".topk", real_ids);
    }

    std::cout << "W (first " << std::min<uint32_t>(ef, 10) << "):";
    for (uint32_t j = 0; j < std::min<uint32_t>(ef, 10); j++) {
        std::cout << " (" << res.w_ref[j].first << "," << res.w_open[j] << ")";
    }
    std::cout << std::endl;
    std::cout << "[Verified] every opened value matches the plaintext reference" << std::endl;
    PrintCost("[Visited check]", res.access);
    PrintCost("[Distance+update]", res.update);
    Cost total = res.access;
    total.Add(res.update);
    PrintTotals(total);
    return 0;
}
