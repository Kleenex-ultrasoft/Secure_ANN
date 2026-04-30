#include "hnsecw_single_b2a.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/sharing/sharing.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

struct OnlineDelta {
    double ms = 0.0;
    uint64_t bytes = 0;
};

static OnlineDelta ExecCircuitDelta(ABYParty* party) {
    // ABY timing/comm counters are per-execution (overwritten on each ExecCircuit).
    party->ExecCircuit();
    return {party->GetTiming(P_ONLINE),
            party->GetSentData(P_ONLINE) + party->GetReceivedData(P_ONLINE)};
}

static void ResetPartyForReuse(ABYParty* party) {
    // Reset circuits/sharings while keeping the existing connection and OT state.
    party->Reset();
}

static inline uint32_t floor_log2(uint32_t n) {
    if (n == 0) return 0;
    uint32_t pos = 0;
    if (n >= 1U << 16) { n >>= 16; pos += 16; }
    if (n >= 1U << 8)  { n >>= 8;  pos += 8; }
    if (n >= 1U << 4)  { n >>= 4;  pos += 4; }
    if (n >= 1U << 2)  { n >>= 2;  pos += 2; }
    if (n >= 1U << 1)  {           pos += 1; }
    return pos;
}

static constexpr uint32_t kDummyIdAuto = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t kEntryAuto = std::numeric_limits<uint32_t>::max();

static uint32_t PickDummyId(uint32_t fallback, uint32_t override_id) {
    return (override_id == kDummyIdAuto) ? fallback : override_id;
}

static bool IsDummyId(uint32_t id, uint32_t real_limit, uint32_t override_id) {
    if (override_id != kDummyIdAuto) {
        return id == override_id;
    }
    return id >= real_limit;
}

static inline uint32_t ceil_log2_u32(uint32_t n) {
    if (n <= 1) {
        return 0;
    }
    uint32_t k = 0;
    uint32_t p = 1;
    while (p < n) {
        p <<= 1;
        k++;
    }
    return k;
}

static inline uint32_t next_power_of_two(uint32_t n) {
    if (n <= 1) return 1;
    uint32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static uint32_t LoadEntryFromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open entry file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size >= static_cast<std::streamoff>(sizeof(uint64_t))) {
        uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(uint64_t));
        return static_cast<uint32_t>(v);
    }
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(uint32_t));
    return v;
}

static void WriteEntryToFile(const std::string& path, uint32_t entry) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open entry_out: " + path);
    }
    out.write(reinterpret_cast<const char*>(&entry), sizeof(uint32_t));
}

static share* ORReduceStreamedEQ(share* x_b, const std::vector<share*>& visited_b,
                                 BooleanCircuit* circ) {
    std::vector<share*> levels;
    levels.reserve(32);
    for (size_t j = 0; j < visited_b.size(); ++j) {
        share* eq = circ->PutEQGate(x_b, visited_b[j]);
        size_t k = 0;
        while (k < levels.size() && levels[k] != nullptr) {
            eq = circ->PutORGate(levels[k], eq);
            levels[k] = nullptr;
            ++k;
        }
        if (k == levels.size()) {
            levels.push_back(eq);
        } else {
            levels[k] = eq;
        }
    }

    share* acc = nullptr;
    for (share* s : levels) {
        if (!s) {
            continue;
        }
        acc = acc ? circ->PutORGate(acc, s) : s;
    }
    if (!acc) {
        acc = circ->PutCONSGate((uint64_t)0, 1);
    }
    return acc;
}

static uint32_t PackYaoValue(share* y, uint32_t bitlen, BooleanCircuit* yao) {
    std::vector<uint32_t> bits(bitlen);
    for (uint32_t l = 0; l < bitlen; l++) {
        bits[l] = y->get_wire_id(l);
    }
    return yao->PutCombinerGate(bits);
}

static std::vector<uint32_t> CondSwap(uint32_t a, uint32_t b, uint32_t s, BooleanCircuit* circ) {
    std::vector<uint32_t> avec(1, a);
    std::vector<uint32_t> bvec(1, b);
    std::vector<uint32_t> out(2);
    std::vector<std::vector<uint32_t>> temp = circ->PutCondSwapGate(avec, bvec, s, true);
    out[0] = temp[0][0];
    out[1] = temp[1][0];
    return out;
}

static void BitonicMergePairs(
    std::vector<uint32_t>& keys,
    std::vector<uint32_t>& payloads,
    uint32_t key_bitlen,
    uint32_t payload_bitlen,
    BooleanCircuit* circ) {

    (void)payload_bitlen;
    uint32_t seqsize = keys.size();
    if (seqsize <= 1) return;
    assert(payloads.size() == seqsize);

    std::vector<uint32_t> c_keys = keys;
    std::vector<uint32_t> c_payloads = payloads;

    uint32_t selbitsvec;
    uint32_t i, k, ctr;
    int32_t j;

    std::vector<uint32_t> compa(seqsize / 2);
    std::vector<uint32_t> compb(seqsize / 2);
    std::vector<uint32_t> selbits;
    std::vector<uint32_t> temp;
    std::vector<uint32_t> tempcmpveca(key_bitlen);
    std::vector<uint32_t> tempcmpvecb(key_bitlen);
    std::vector<uint32_t> parenta(seqsize / 2);
    std::vector<uint32_t> parentb(seqsize / 2);

    for (i = 1U << floor_log2(seqsize - 1); i > 0; i >>= 1) {
        ctr = 0;
        for (j = (int32_t)seqsize - 1; j >= 0; j -= (int32_t)(2 * i)) {
            for (k = 0; k < i && j - (int32_t)i - (int32_t)k >= 0; k++) {
                compa[ctr] = (uint32_t)(j - (int32_t)i - (int32_t)k);
                compb[ctr] = (uint32_t)(j - (int32_t)k);
                ctr++;
            }
        }

        for (uint32_t l = 0; l < key_bitlen; l++) {
            for (k = 0; k < ctr; k++) {
                parenta[k] = c_keys[compa[k]];
                parentb[k] = c_keys[compb[k]];
            }
            tempcmpveca[l] = circ->PutCombineAtPosGate(parenta, l);
            tempcmpvecb[l] = circ->PutCombineAtPosGate(parentb, l);
        }

        selbitsvec = circ->PutGTGate(tempcmpveca, tempcmpvecb);
        selbits = circ->PutSplitterGate(selbitsvec);

        for (k = 0; k < ctr; k++) {
            temp = CondSwap(c_keys[compa[k]], c_keys[compb[k]], selbits[k], circ);
            c_keys[compa[k]] = temp[0];
            c_keys[compb[k]] = temp[1];

            temp = CondSwap(c_payloads[compa[k]], c_payloads[compb[k]], selbits[k], circ);
            c_payloads[compa[k]] = temp[0];
            c_payloads[compb[k]] = temp[1];
        }
    }

    keys.swap(c_keys);
    payloads.swap(c_payloads);
}

static void BitonicFullSortPairs(
    std::vector<uint32_t>& keys,
    std::vector<uint32_t>& payloads,
    uint32_t key_bitlen,
    uint32_t payload_bitlen,
    BooleanCircuit* circ) {

    uint32_t n = keys.size();
    if (n <= 1) return;
    assert(payloads.size() == n);

    if ((n & (n - 1)) != 0) {
        uint32_t orig_n = n;
        uint32_t padded = next_power_of_two(n);

        share* max_val = circ->PutCONSGate((uint64_t)-1, key_bitlen);
        uint32_t max_wire = PackYaoValue(max_val, key_bitlen, circ);

        share* zero_payload = circ->PutCONSGate((uint64_t)0, payload_bitlen);
        uint32_t zero_wire = PackYaoValue(zero_payload, payload_bitlen, circ);

        while (keys.size() < padded) {
            keys.push_back(max_wire);
            payloads.push_back(zero_wire);
        }

        BitonicFullSortPairs(keys, payloads, key_bitlen, payload_bitlen, circ);

        keys.resize(orig_n);
        payloads.resize(orig_n);
        return;
    }

    std::vector<uint32_t> lower_keys(keys.begin(), keys.begin() + n / 2);
    std::vector<uint32_t> upper_keys(keys.begin() + n / 2, keys.end());
    std::vector<uint32_t> lower_payloads(payloads.begin(), payloads.begin() + n / 2);
    std::vector<uint32_t> upper_payloads(payloads.begin() + n / 2, payloads.end());

    BitonicFullSortPairs(lower_keys, lower_payloads, key_bitlen, payload_bitlen, circ);
    BitonicFullSortPairs(upper_keys, upper_payloads, key_bitlen, payload_bitlen, circ);

    std::reverse(upper_keys.begin(), upper_keys.end());
    std::reverse(upper_payloads.begin(), upper_payloads.end());

    std::vector<uint32_t> bitonic_keys;
    std::vector<uint32_t> bitonic_payloads;
    bitonic_keys.insert(bitonic_keys.end(), lower_keys.begin(), lower_keys.end());
    bitonic_keys.insert(bitonic_keys.end(), upper_keys.begin(), upper_keys.end());
    bitonic_payloads.insert(bitonic_payloads.end(), lower_payloads.begin(), lower_payloads.end());
    bitonic_payloads.insert(bitonic_payloads.end(), upper_payloads.begin(), upper_payloads.end());

    BitonicMergePairs(bitonic_keys, bitonic_payloads, key_bitlen, payload_bitlen, circ);

    keys.swap(bitonic_keys);
    payloads.swap(bitonic_payloads);
}

static void PlainBitonicMergePairs(std::vector<std::pair<uint64_t, uint64_t>>& pairs) {
    uint32_t seqsize = pairs.size();
    if (seqsize <= 1) return;

    std::vector<std::pair<uint64_t, uint64_t>> c = pairs;

    uint32_t i, k, ctr;
    int32_t j;

    std::vector<uint32_t> compa(seqsize / 2);
    std::vector<uint32_t> compb(seqsize / 2);

    for (i = 1U << floor_log2(seqsize - 1); i > 0; i >>= 1) {
        ctr = 0;
        for (j = (int32_t)seqsize - 1; j >= 0; j -= (int32_t)(2 * i)) {
            for (k = 0; k < i && j - (int32_t)i - (int32_t)k >= 0; k++) {
                compa[ctr] = (uint32_t)(j - (int32_t)i - (int32_t)k);
                compb[ctr] = (uint32_t)(j - (int32_t)k);
                ctr++;
            }
        }

        for (k = 0; k < ctr; k++) {
            uint32_t a = compa[k];
            uint32_t b = compb[k];
            if (c[a].first > c[b].first) {
                std::swap(c[a], c[b]);
            }
        }
    }

    pairs.swap(c);
}

static void PlainBitonicFullSortPairs(std::vector<std::pair<uint64_t, uint64_t>>& pairs, uint32_t key_bitlen) {
    uint32_t n = pairs.size();
    if (n <= 1) return;

    if ((n & (n - 1)) != 0) {
        uint32_t orig_n = n;
        uint32_t padded = next_power_of_two(n);
        uint64_t max_key = (key_bitlen >= 64) ? ~0ULL : ((1ULL << key_bitlen) - 1ULL);

        while (pairs.size() < padded) {
            pairs.push_back(std::make_pair(max_key, 0));
        }

        PlainBitonicFullSortPairs(pairs, key_bitlen);

        pairs.resize(orig_n);
        return;
    }

    std::vector<std::pair<uint64_t, uint64_t>> lower(pairs.begin(), pairs.begin() + n / 2);
    std::vector<std::pair<uint64_t, uint64_t>> upper(pairs.begin() + n / 2, pairs.end());

    PlainBitonicFullSortPairs(lower, key_bitlen);
    PlainBitonicFullSortPairs(upper, key_bitlen);

    std::reverse(upper.begin(), upper.end());

    std::vector<std::pair<uint64_t, uint64_t>> bitonic;
    bitonic.insert(bitonic.end(), lower.begin(), lower.end());
    bitonic.insert(bitonic.end(), upper.begin(), upper.end());

    PlainBitonicMergePairs(bitonic);
    pairs.swap(bitonic);
}

struct ExecAResult {
    std::vector<uint32_t> id_prime;
    std::vector<uint8_t> hit_share;
    double online_ms = 0.0;
    uint64_t online_bytes = 0;
};

struct ExecBResult {
    std::vector<uint32_t> C_dist;
    std::vector<uint32_t> C_id;
    std::vector<uint32_t> W_dist;
    std::vector<uint32_t> W_id;
    double online_ms = 0.0;
    uint64_t online_bytes = 0;
};

static void PrintPartyStats(ABYParty* party, const std::vector<Sharing*>& sharings,
                            const std::string& label, bool show_bool, bool show_arith,
                            bool show_yao) {
    uint64_t comm_total = party->GetSentData(P_TOTAL) + party->GetReceivedData(P_TOTAL);
    uint64_t comm_online = party->GetSentData(P_ONLINE) + party->GetReceivedData(P_ONLINE);
    uint64_t comm_setup = party->GetSentData(P_SETUP) + party->GetReceivedData(P_SETUP);

    std::cout << label << " timings (ms): setup=" << party->GetTiming(P_SETUP)
              << " online=" << party->GetTiming(P_ONLINE)
              << " total=" << party->GetTiming(P_TOTAL) << std::endl;
    std::cout << label << " comm (bytes): setup=" << comm_setup
              << " online=" << comm_online
              << " total=" << comm_total << std::endl;

    if (show_bool) {
        std::cout << label << " BOOL ops=" << sharings[S_BOOL]->GetNumNonLinearOperations()
                  << " depth=" << sharings[S_BOOL]->GetMaxCommunicationRounds() << std::endl;
    }
    if (show_arith) {
        std::cout << label << " ARITH ops=" << sharings[S_ARITH]->GetNumNonLinearOperations()
                  << " depth=" << sharings[S_ARITH]->GetMaxCommunicationRounds() << std::endl;
    }
    if (show_yao) {
        std::cout << label << " YAO ops=" << sharings[S_YAO]->GetNumNonLinearOperations()
                  << " depth=" << sharings[S_YAO]->GetMaxCommunicationRounds() << std::endl;
    }
}

static uint32_t NextId(uint32_t& state, uint32_t id_mask) {
    state = state * 1664525u + 1013904223u;
    return state & id_mask;
}

// Real-data tables loaded from HNSECW_DATA_FILE (same schema as single_b2y).
struct RealDataTables {
    bool loaded = false;
    uint32_t N = 0;
    uint32_t D = 0;
    uint32_t M = 0;
    uint32_t num_queries = 0;
    std::vector<std::vector<uint32_t>> graph;
    std::vector<std::vector<uint32_t>> vec;
    std::vector<std::vector<uint32_t>> query;
};

static RealDataTables g_real_data;

static void LoadRealDataIfRequested() {
    if (g_real_data.loaded) return;
    const char* env = std::getenv("HNSECW_DATA_FILE");
    if (!env || std::string(env).empty()) return;
    std::ifstream in(env, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[HNSECW_DATA_FILE] failed to open: " << env << std::endl;
        return;
    }
    uint32_t header[4];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    g_real_data.N = header[0];
    g_real_data.D = header[1];
    g_real_data.M = header[2];
    g_real_data.num_queries = header[3];
    g_real_data.graph.assign(g_real_data.N, std::vector<uint32_t>(g_real_data.M));
    for (uint32_t i = 0; i < g_real_data.N; ++i) {
        in.read(reinterpret_cast<char*>(g_real_data.graph[i].data()),
                g_real_data.M * sizeof(uint32_t));
    }
    g_real_data.vec.assign(g_real_data.N, std::vector<uint32_t>(g_real_data.D));
    for (uint32_t i = 0; i < g_real_data.N; ++i) {
        in.read(reinterpret_cast<char*>(g_real_data.vec[i].data()),
                g_real_data.D * sizeof(uint32_t));
    }
    g_real_data.query.assign(g_real_data.num_queries,
                             std::vector<uint32_t>(g_real_data.D));
    for (uint32_t q = 0; q < g_real_data.num_queries; ++q) {
        in.read(reinterpret_cast<char*>(g_real_data.query[q].data()),
                g_real_data.D * sizeof(uint32_t));
    }
    g_real_data.loaded = true;
    std::cerr << "[HNSECW_DATA_FILE] loaded N=" << g_real_data.N
              << " D=" << g_real_data.D
              << " M=" << g_real_data.M
              << " Q=" << g_real_data.num_queries << std::endl;
}

static std::vector<uint32_t> GenerateNeighbors(uint32_t u, uint32_t t, uint32_t M,
                                               uint32_t real_node_limit, uint32_t avoid,
                                               uint32_t dummy_id_override) {
    LoadRealDataIfRequested();
    if (g_real_data.loaded && u < g_real_data.N && M == g_real_data.M) {
        return g_real_data.graph[u];
    }
    if (IsDummyId(u, real_node_limit, dummy_id_override)) {
        return std::vector<uint32_t>(M, u);
    }
    std::vector<uint32_t> out;
    out.reserve(M);
    std::unordered_set<uint32_t> used;
    uint32_t state = u ^ (t * 0x9e3779b9u) ^ 0xa5a5a5a5u;
    while (out.size() < M) {
        uint32_t v = NextId(state, 0xFFFFFFFFu);
        if (real_node_limit > 0) {
            v %= real_node_limit;
        }
        if (v == avoid) {
            continue;
        }
        if (used.insert(v).second) {
            out.push_back(v);
        }
    }
    return out;
}

static std::vector<uint32_t> MakeVectorForId(uint32_t id, uint32_t D, uint32_t bitlen) {
    LoadRealDataIfRequested();
    if (g_real_data.loaded && id < g_real_data.N && D == g_real_data.D) {
        return g_real_data.vec[id];
    }
    std::vector<uint32_t> out(D, 0);
    uint64_t bound = 1;
    if (bitlen >= 2 && bitlen < 32) {
        bound = 1ULL << (bitlen - 1);
    } else if (bitlen >= 32) {
        bound = 1ULL << 31;
    }
    for (uint32_t j = 0; j < D; j++) {
        uint64_t v = (uint64_t)id * 2654435761u + (uint64_t)j * 2246822519u;
        out[j] = (uint32_t)(v % bound);
    }
    return out;
}

static std::vector<std::vector<uint32_t>> FetchVectors(
    const std::vector<uint32_t>& id_prime,
    uint32_t real_node_limit,
    uint32_t D,
    uint32_t bitlen,
    uint32_t dummy_id_override) {

    std::vector<std::vector<uint32_t>> out(id_prime.size(), std::vector<uint32_t>(D, 0));
    for (size_t i = 0; i < id_prime.size(); i++) {
        if (IsDummyId(id_prime[i], real_node_limit, dummy_id_override)) {
            continue;
        }
        out[i] = MakeVectorForId(id_prime[i], D, bitlen);
    }
    return out;
}

static ExecAResult RunExecA(
    ABYParty* party,
    e_role role,
    uint32_t M,
    uint32_t id_bitlen,
    const std::vector<uint32_t>& neigh_ids,
    const std::vector<uint32_t>& dummy_id_plain,
    const std::vector<uint32_t>& visited_plain,
    uint32_t visited_len_eff,
    uint32_t debug_tag,
    const std::string& label) {

    ResetPartyForReuse(party);
    (void)debug_tag;

    std::vector<Sharing*>& sharings = party->GetSharings();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
    ArithmeticCircuit* arith_circ = (ArithmeticCircuit*) sharings[S_ARITH]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    const e_role OWNER = SERVER;

    assert(visited_len_eff <= visited_plain.size());
    std::vector<share*> cleanup;
    // Conservative reserve to avoid frequent reallocations in O(M * visited_len_eff) loop.
    cleanup.reserve((size_t)M * (size_t)visited_len_eff * 6 + 2048);
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    std::vector<share*> visited_consts(visited_len_eff);
    for (uint32_t i = 0; i < visited_len_eff; i++) {
        visited_consts[i] = track(bool_circ->PutCONSGate(visited_plain[i], id_bitlen));
    }

    std::vector<share*> id_b_shares(M);
    for (uint32_t i = 0; i < M; i++) {
        id_b_shares[i] = track(bool_circ->PutINGate(
            (role == OWNER ? (uint64_t)neigh_ids[i] : 0),
            id_bitlen,
            OWNER));
    }

    std::vector<share*> id_prime_out(M);
    std::vector<share*> hit_masked_out(M);
    std::vector<uint8_t> server_share(M, 0);

    // Sum is in [0, visited_len_eff], so only low-k bits are needed for nonzero test.
    uint32_t arith_bitlen = arith_circ->GetShareBitLen();
    uint32_t sum_bits = ceil_log2_u32(visited_len_eff + 1);
    if (sum_bits == 0) {
        sum_bits = 1;
    }

    for (uint32_t i = 0; i < M; i++) {
        share* id_b = id_b_shares[i];
        share* sum_a = track(arith_circ->PutCONSGate((uint64_t)0, arith_bitlen));
        for (uint32_t j = 0; j < visited_len_eff; j++) {
            share* eq_b = track(bool_circ->PutEQGate(id_b, visited_consts[j]));
            share* eq_a = track(arith_circ->PutB2AGate(eq_b));
            sum_a = track(arith_circ->PutADDGate(sum_a, eq_a));
        }

        share* sum_y_full = track(yao_circ->PutA2YGate(sum_a));
        // Use only low-k bits of sum for the nonzero check to cut Yao EQ width.
        boolshare* sum_y_k = new boolshare(sum_bits, yao_circ);
        cleanup.push_back(sum_y_k);
        for (uint32_t b = 0; b < sum_bits; b++) {
            sum_y_k->set_wire_id(b, sum_y_full->get_wire_id(b));
        }
        share* zero_y = track(yao_circ->PutCONSGate((uint64_t)0, sum_bits));
        share* is_zero = track(yao_circ->PutEQGate(sum_y_k, zero_y));
        share* hit_y = track(yao_circ->PutINVGate(is_zero));

        if (role == SERVER) {
            server_share[i] = (uint8_t)(rand() & 1U);
        }
        share* r_y = track(yao_circ->PutINGate(
            (role == SERVER ? (uint64_t)server_share[i] : 0),
            1,
            SERVER));
        share* hit_masked_y = track(yao_circ->PutXORGate(hit_y, r_y));

        share* id_y = track(yao_circ->PutINGate(
            (role == OWNER ? (uint64_t)neigh_ids[i] : 0),
            id_bitlen,
            OWNER));
        share* dummy_y = track(yao_circ->PutINGate(
            (role == OWNER ? (uint64_t)dummy_id_plain[i] : 0),
            id_bitlen,
            OWNER));

        // PutMUXGate selects first arg when sel=1, so hit->dummy, miss->id.
        share* id_sel_y = track(yao_circ->PutMUXGate(dummy_y, id_y, hit_y));

        id_prime_out[i] = track(yao_circ->PutOUTGate(id_sel_y, ALL));
        hit_masked_out[i] = track(yao_circ->PutOUTGate(hit_masked_y, CLIENT));
    }

    std::cout << "\n" << label << " Executing" << std::endl;
    OnlineDelta delta = ExecCircuitDelta(party);
    std::cout << label << " Executed" << std::endl;

    ExecAResult res;
    res.id_prime.resize(M);
    res.hit_share.resize(M);
    res.online_ms = delta.ms;
    res.online_bytes = delta.bytes;
    for (uint32_t i = 0; i < M; i++) {
        res.id_prime[i] = (uint32_t)id_prime_out[i]->get_clear_value<uint64_t>();
        if (role == SERVER) {
            res.hit_share[i] = server_share[i];
        } else {
            res.hit_share[i] = (uint8_t)(hit_masked_out[i]->get_clear_value<uint64_t>() & 1ULL);
        }
    }

    if (role == SERVER) {
        std::cout << "\n" << label << " Completed membership + ID reveal" << std::endl;
        PrintPartyStats(party, sharings, label, true, false, false);
    }

    for (share* s : cleanup) {
        delete s;
    }
    return res;
}

static ExecBResult RunExecB(
    ABYParty* party,
    e_role role,
    uint32_t M, uint32_t D, uint32_t L_C, uint32_t L_W,
    uint32_t bitlen, uint32_t id_bitlen, uint32_t inner_prod_bitlen,
    uint32_t debug_tag,
    uint32_t iter,
    const std::vector<uint32_t>& query_plain,
    const std::vector<std::vector<uint32_t>>& fetched_vectors,
    const std::vector<uint8_t>& hit_share,
    const std::vector<uint32_t>& id_prime_plain,
    uint32_t real_node_limit,
    const std::vector<uint32_t>& C_dist_plain,
    const std::vector<uint32_t>& C_id_plain,
    const std::vector<uint32_t>& W_dist_plain,
    const std::vector<uint32_t>& W_id_plain,
    uint64_t md_key,
    const std::string& label) {

    ResetPartyForReuse(party);

    std::vector<Sharing*>& sharings = party->GetSharings();
    ArithmeticCircuit* arith_circ = (ArithmeticCircuit*) sharings[S_ARITH]->GetCircuitBuildRoutine();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    const e_role OWNER = SERVER;

    std::vector<share*> cleanup;
    cleanup.reserve(16 * M + L_C + L_W);
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    std::vector<uint32_t> zeros_query(D, 0);
    std::vector<uint32_t> zeros_vec(D, 0);

    uint32_t* query_ptr = (role == OWNER
        ? const_cast<uint32_t*>(query_plain.data())
        : zeros_query.data());
    share* s_query = track(arith_circ->PutSIMDINGate(D, query_ptr, bitlen, OWNER));

    share** inner_products = new share*[M];
    for (uint32_t i = 0; i < M; i++) {
        uint32_t* vec_ptr = (role == OWNER
            ? const_cast<uint32_t*>(fetched_vectors[i].data())
            : zeros_vec.data());
        share* s_vec = track(arith_circ->PutSIMDINGate(D, vec_ptr, bitlen, OWNER));

        // Squared L2 distance: (v - q)^2 element-wise, then sum.
        // See hnsecw_single_b2y.cpp for the rationale.
        share* s_diff = track(arith_circ->PutSUBGate(s_vec, s_query));
        share* s_products = track(arith_circ->PutMULGate(s_diff, s_diff));
        s_products = track(arith_circ->PutSplitterGate(s_products));

        std::vector<uint32_t> sum_wires;
        sum_wires.reserve(D);
        for (uint32_t j = 0; j < D; j++) {
            sum_wires.push_back(s_products->get_wire_id(j));
        }

        while (sum_wires.size() > 1) {
            std::vector<uint32_t> next_level;
            for (uint32_t j = 0; j + 1 < sum_wires.size(); j += 2) {
                next_level.push_back(arith_circ->PutADDGate(sum_wires[j], sum_wires[j + 1]));
            }
            if (sum_wires.size() % 2 == 1) {
                next_level.push_back(sum_wires.back());
            }
            sum_wires.swap(next_level);
        }

        inner_products[i] = new arithshare(arith_circ);
        cleanup.push_back(inner_products[i]);
        inner_products[i]->set_wire_id(0, sum_wires[0]);
    }

    share* md_a = track(arith_circ->PutCONSGate(md_key, inner_prod_bitlen));

#ifdef DBG_HIT
    uint32_t dbg_limit = 0;
    if (debug_tag) {
        dbg_limit = std::min(M, 8U);
    }
    std::vector<share*> dbg_hit_b_out(dbg_limit, nullptr);
    std::vector<share*> dbg_hit_a_out(dbg_limit, nullptr);
    std::vector<share*> dbg_dist_out(dbg_limit, nullptr);
#endif

    std::vector<share*> cand_dist_y(M);
    std::vector<share*> cand_id_y(M);
    for (uint32_t i = 0; i < M; i++) {
        assert((hit_share[i] & 1U) == hit_share[i]);
        share* dist_a = inner_products[i];
        share* hit_server = track(bool_circ->PutINGate(
            (role == SERVER ? (uint64_t)hit_share[i] : 0),
            1,
            SERVER));
        share* hit_client = track(bool_circ->PutINGate(
            (role == CLIENT ? (uint64_t)hit_share[i] : 0),
            1,
            CLIENT));
        share* hit_b = track(bool_circ->PutXORGate(hit_server, hit_client));
        share* hit_a = track(arith_circ->PutB2AGate(hit_b));

        share* diff = track(arith_circ->PutSUBGate(md_a, dist_a));
        share* prod = track(arith_circ->PutMULGate(hit_a, diff));
        share* dist_sel_a = track(arith_circ->PutADDGate(dist_a, prod));
        share* dist_sel_y = track(yao_circ->PutA2YGate(dist_sel_a));

        share* id_y = track(yao_circ->PutCONSGate(id_prime_plain[i], id_bitlen));

        cand_dist_y[i] = dist_sel_y;
        cand_id_y[i] = id_y;

#ifdef DBG_HIT
        if (debug_tag && i < dbg_limit) {
            dbg_hit_b_out[i] = track(bool_circ->PutOUTGate(hit_b, CLIENT));
            dbg_hit_a_out[i] = track(arith_circ->PutOUTGate(hit_a, CLIENT));
            dbg_dist_out[i] = track(arith_circ->PutOUTGate(dist_sel_a, CLIENT));
        }
#endif
    }

    share** C_dist_shares = new share*[L_C];
    share** C_id_shares = new share*[L_C];
    share** W_dist_shares = new share*[L_W];
    share** W_id_shares = new share*[L_W];

    for (uint32_t i = 0; i < L_C; i++) {
        C_dist_shares[i] = track(arith_circ->PutINGate(
            (role == OWNER ? (uint64_t)C_dist_plain[i] : 0),
            inner_prod_bitlen,
            OWNER));
        C_id_shares[i] = track(arith_circ->PutINGate(
            (role == OWNER ? (uint64_t)C_id_plain[i] : 0),
            id_bitlen,
            OWNER));
    }
    for (uint32_t i = 0; i < L_W; i++) {
        W_dist_shares[i] = track(arith_circ->PutINGate(
            (role == OWNER ? (uint64_t)W_dist_plain[i] : 0),
            inner_prod_bitlen,
            OWNER));
        W_id_shares[i] = track(arith_circ->PutINGate(
            (role == OWNER ? (uint64_t)W_id_plain[i] : 0),
            id_bitlen,
            OWNER));
    }

    uint32_t total_C = L_C + M;
    uint32_t total_W = L_W + M;
    uint32_t tag_bits = 0;
    uint32_t payload_bitlen = id_bitlen;
    if (debug_tag) {
        tag_bits = (M + 1 <= 1) ? 1U : (uint32_t)ceil(log2((double)(M + 1)));
        payload_bitlen = id_bitlen + tag_bits;
    }

    auto pack_payload = [&](share* id_y, uint32_t tag) -> uint32_t {
        if (!debug_tag) {
            return PackYaoValue(id_y, id_bitlen, yao_circ);
        }
        std::vector<uint32_t> bits(payload_bitlen);
        for (uint32_t l = 0; l < id_bitlen; l++) {
            bits[l] = id_y->get_wire_id(l);
        }
        share* tag_const = track(yao_circ->PutCONSGate(tag, tag_bits));
        for (uint32_t l = 0; l < tag_bits; l++) {
            bits[id_bitlen + l] = tag_const->get_wire_id(l);
        }
        return yao_circ->PutCombinerGate(bits);
    };

    std::vector<uint32_t> cand_key_wires(M);
    std::vector<uint32_t> cand_pay_wires(M);
    for (uint32_t i = 0; i < M; i++) {
        cand_key_wires[i] = PackYaoValue(cand_dist_y[i], inner_prod_bitlen, yao_circ);
        cand_pay_wires[i] = pack_payload(cand_id_y[i], i + 1);
    }
    if (iter > 0) {
        BitonicFullSortPairs(cand_key_wires, cand_pay_wires, inner_prod_bitlen, payload_bitlen, yao_circ);
    }

    std::vector<share*> C_dist_y(L_C);
    std::vector<share*> C_id_y(L_C);
    std::vector<share*> W_dist_y(L_W);
    std::vector<share*> W_id_y(L_W);

    for (uint32_t i = 0; i < L_C; i++) {
        C_dist_y[i] = track(yao_circ->PutA2YGate(C_dist_shares[i]));
        C_id_y[i] = track(yao_circ->PutA2YGate(C_id_shares[i]));
    }
    for (uint32_t i = 0; i < L_W; i++) {
        W_dist_y[i] = track(yao_circ->PutA2YGate(W_dist_shares[i]));
        W_id_y[i] = track(yao_circ->PutA2YGate(W_id_shares[i]));
    }

    std::vector<uint32_t> C_key_wires;
    std::vector<uint32_t> C_pay_wires;
    std::vector<uint32_t> W_key_wires;
    std::vector<uint32_t> W_pay_wires;

    // Always take the full-sort path; merge branch consumes more
    // arithmetic OT triples and triggers IKNP-OT pool issues at
    // fashion scale.  See hnsecw_single_b2y.cpp for full context.
    if (true /* was: iter == 0 */) {
        C_key_wires.resize(total_C);
        C_pay_wires.resize(total_C);
        W_key_wires.resize(total_W);
        W_pay_wires.resize(total_W);

        for (uint32_t i = 0; i < L_C; i++) {
            C_key_wires[i] = PackYaoValue(C_dist_y[i], inner_prod_bitlen, yao_circ);
            C_pay_wires[i] = pack_payload(C_id_y[i], 0);
        }
        for (uint32_t i = 0; i < M; i++) {
            C_key_wires[L_C + i] = cand_key_wires[i];
            C_pay_wires[L_C + i] = cand_pay_wires[i];
        }

        for (uint32_t i = 0; i < L_W; i++) {
            W_key_wires[i] = PackYaoValue(W_dist_y[i], inner_prod_bitlen, yao_circ);
            W_pay_wires[i] = pack_payload(W_id_y[i], 0);
        }
        for (uint32_t i = 0; i < M; i++) {
            W_key_wires[L_W + i] = cand_key_wires[i];
            W_pay_wires[L_W + i] = cand_pay_wires[i];
        }

        // Warm-start: full sort to handle initially unsorted C/W.
        BitonicFullSortPairs(C_key_wires, C_pay_wires, inner_prod_bitlen, payload_bitlen, yao_circ);
        BitonicFullSortPairs(W_key_wires, W_pay_wires, inner_prod_bitlen, payload_bitlen, yao_circ);
    } else {
        std::vector<uint32_t> C_old_keys(L_C);
        std::vector<uint32_t> C_old_payloads(L_C);
        std::vector<uint32_t> W_old_keys(L_W);
        std::vector<uint32_t> W_old_payloads(L_W);

        for (uint32_t i = 0; i < L_C; i++) {
            C_old_keys[i] = PackYaoValue(C_dist_y[i], inner_prod_bitlen, yao_circ);
            C_old_payloads[i] = pack_payload(C_id_y[i], 0);
        }
        for (uint32_t i = 0; i < L_W; i++) {
            W_old_keys[i] = PackYaoValue(W_dist_y[i], inner_prod_bitlen, yao_circ);
            W_old_payloads[i] = pack_payload(W_id_y[i], 0);
        }

        share* max_key_val = track(yao_circ->PutCONSGate((uint64_t)-1, inner_prod_bitlen));
        uint32_t max_key_wire = PackYaoValue(max_key_val, inner_prod_bitlen, yao_circ);
        share* zero_payload = track(yao_circ->PutCONSGate((uint64_t)0, payload_bitlen));
        uint32_t zero_payload_wire = PackYaoValue(zero_payload, payload_bitlen, yao_circ);

        auto merge_sorted_topk =
            [&](const std::vector<uint32_t>& a_keys,
                const std::vector<uint32_t>& a_payloads,
                const std::vector<uint32_t>& b_keys,
                const std::vector<uint32_t>& b_payloads,
                uint32_t keep,
                std::vector<uint32_t>& out_keys,
                std::vector<uint32_t>& out_payloads) {
                // Lay out as [a_asc | INF padding | reverse(b_asc)] -- a valid
                // bitonic input.  INF padding sits at the peak between the
                // ascending half and the reversed (descending) half so the
                // single-peak invariant holds for any |a|+|b|.
                size_t total = a_keys.size() + b_keys.size();
                size_t padded = next_power_of_two((uint32_t)total);
                size_t pad_count = padded - total;
                out_keys.clear();
                out_payloads.clear();
                out_keys.reserve(padded);
                out_payloads.reserve(padded);
                out_keys.insert(out_keys.end(), a_keys.begin(), a_keys.end());
                out_payloads.insert(out_payloads.end(), a_payloads.begin(), a_payloads.end());
                for (size_t i = 0; i < pad_count; i++) {
                    out_keys.push_back(max_key_wire);
                    out_payloads.push_back(zero_payload_wire);
                }
                for (uint32_t idx = 0; idx < b_keys.size(); idx++) {
                    size_t r = b_keys.size() - 1 - idx;
                    out_keys.push_back(b_keys[r]);
                    out_payloads.push_back(b_payloads[r]);
                }
                BitonicMergePairs(out_keys, out_payloads, inner_prod_bitlen, payload_bitlen, yao_circ);
                if (out_keys.size() > keep) {
                    out_keys.resize(keep);
                    out_payloads.resize(keep);
                }
            };

        merge_sorted_topk(C_old_keys, C_old_payloads,
                          cand_key_wires, cand_pay_wires,
                          total_C, C_key_wires, C_pay_wires);
        merge_sorted_topk(W_old_keys, W_old_payloads,
                          cand_key_wires, cand_pay_wires,
                          total_W, W_key_wires, W_pay_wires);
    }

    share** C_key_out = new share*[total_C];
    share** C_pay_out = new share*[total_C];
    share** W_key_out = new share*[total_W];
    share** W_pay_out = new share*[total_W];

    for (uint32_t i = 0; i < total_C; i++) {
        share* tmp = new boolshare(1, yao_circ);
        cleanup.push_back(tmp);
        tmp->set_wire_id(0, C_key_wires[i]);
        C_key_out[i] = track(yao_circ->PutOUTGate(tmp, ALL));
    }
    for (uint32_t i = 0; i < total_C; i++) {
        share* tmp = new boolshare(1, yao_circ);
        cleanup.push_back(tmp);
        tmp->set_wire_id(0, C_pay_wires[i]);
        C_pay_out[i] = track(yao_circ->PutOUTGate(tmp, ALL));
    }
    for (uint32_t i = 0; i < total_W; i++) {
        share* tmp = new boolshare(1, yao_circ);
        cleanup.push_back(tmp);
        tmp->set_wire_id(0, W_key_wires[i]);
        W_key_out[i] = track(yao_circ->PutOUTGate(tmp, ALL));
    }
    for (uint32_t i = 0; i < total_W; i++) {
        share* tmp = new boolshare(1, yao_circ);
        cleanup.push_back(tmp);
        tmp->set_wire_id(0, W_pay_wires[i]);
        W_pay_out[i] = track(yao_circ->PutOUTGate(tmp, ALL));
    }

    std::cout << "\n" << label << " Executing" << std::endl;
    OnlineDelta delta = ExecCircuitDelta(party);
    std::cout << label << " Executed" << std::endl;

    if (role == SERVER) {
        PrintPartyStats(party, sharings, label, false, true, true);
    }

    ExecBResult res;
    res.C_dist.resize(L_C);
    res.C_id.resize(L_C);
    res.W_dist.resize(L_W);
    res.W_id.resize(L_W);
    res.online_ms = delta.ms;
    res.online_bytes = delta.bytes;

    std::vector<uint64_t> C_key_results(total_C);
    std::vector<uint64_t> C_pay_results(total_C);
    std::vector<uint64_t> W_key_results(total_W);
    std::vector<uint64_t> W_pay_results(total_W);

    for (uint32_t i = 0; i < total_C; i++) {
        C_key_results[i] = C_key_out[i]->get_clear_value<uint64_t>();
        C_pay_results[i] = C_pay_out[i]->get_clear_value<uint64_t>();
    }
    for (uint32_t i = 0; i < total_W; i++) {
        W_key_results[i] = W_key_out[i]->get_clear_value<uint64_t>();
        W_pay_results[i] = W_pay_out[i]->get_clear_value<uint64_t>();
    }

    uint64_t id_mask64 = (id_bitlen >= 64) ? ~0ULL : ((1ULL << id_bitlen) - 1ULL);
    for (uint32_t i = 0; i < L_C; i++) {
        res.C_dist[i] = (uint32_t)C_key_results[i];
        res.C_id[i] = (uint32_t)(C_pay_results[i] & id_mask64); // strip debug tag bits
    }
    for (uint32_t i = 0; i < L_W; i++) {
        res.W_dist[i] = (uint32_t)W_key_results[i];
        res.W_id[i] = (uint32_t)(W_pay_results[i] & id_mask64);
    }

    if (role == CLIENT) {
#ifdef DBG_HIT
        if (debug_tag && dbg_limit > 0) {
            for (uint32_t i = 0; i < dbg_limit; i++) {
                uint8_t hit_open = (uint8_t)(dbg_hit_b_out[i]->get_clear_value<uint64_t>() & 1ULL);
                uint64_t hit_a_open = dbg_hit_a_out[i]->get_clear_value<uint64_t>();
                uint64_t dist_open = dbg_dist_out[i]->get_clear_value<uint64_t>();
                bool hit_should = (id_prime_plain[i] >= real_node_limit);
                uint64_t inner_sum = 0;
                for (uint32_t j = 0; j < D; j++) {
                    inner_sum += (uint64_t)fetched_vectors[i][j] * query_plain[j];
                }
                uint64_t dist_should = hit_should ? md_key : inner_sum;
                std::cout << "[DBG hit] i=" << i
                          << " hit_open=" << (int)hit_open
                          << " hit_should=" << (int)hit_should
                          << " hit_a=" << hit_a_open
                          << " dist_open=" << dist_open
                          << " dist_should=" << dist_should
                          << " id=" << id_prime_plain[i]
                          << std::endl;
            }
        }
#endif
        std::cout << "Sorted C pairs (first 10): ";
        for (uint32_t i = 0; i < std::min(total_C, 10U); i++) {
            std::cout << "(" << C_key_results[i] << "," << C_pay_results[i] << ") ";
        }
        std::cout << (total_C > 10 ? "..." : "") << std::endl;

        std::cout << "Sorted W pairs (first 10): ";
        for (uint32_t i = 0; i < std::min(total_W, 10U); i++) {
            std::cout << "(" << W_key_results[i] << "," << W_pay_results[i] << ") ";
        }
        std::cout << (total_W > 10 ? "..." : "") << std::endl;

        if (total_C <= 64 && total_W <= 64) {
            std::vector<uint64_t> inner_prod_plain(M);
            for (uint32_t i = 0; i < M; i++) {
                uint64_t sum = 0;
                for (uint32_t j = 0; j < D; j++) {
                    sum += (uint64_t)fetched_vectors[i][j] * query_plain[j];
                }
                inner_prod_plain[i] = sum;
            }

            auto make_payload_plain = [&](uint64_t idv, uint64_t tag) -> uint64_t {
                if (!debug_tag) {
                    return idv;
                }
                return (tag << id_bitlen) | idv;
            };

            std::vector<std::pair<uint64_t, uint64_t>> pairs_c_old;
            std::vector<std::pair<uint64_t, uint64_t>> pairs_w_old;
            std::vector<std::pair<uint64_t, uint64_t>> pairs_cand;
            pairs_c_old.reserve(L_C);
            pairs_w_old.reserve(L_W);
            pairs_cand.reserve(M);

            for (uint32_t i = 0; i < L_C; i++) {
                pairs_c_old.push_back(std::make_pair(
                    (uint64_t)C_dist_plain[i],
                    make_payload_plain((uint64_t)C_id_plain[i], 0)));
            }
            for (uint32_t i = 0; i < L_W; i++) {
                pairs_w_old.push_back(std::make_pair(
                    (uint64_t)W_dist_plain[i],
                    make_payload_plain((uint64_t)W_id_plain[i], 0)));
            }

            for (uint32_t i = 0; i < M; i++) {
                bool hit_debug = (id_prime_plain[i] >= real_node_limit);
                uint64_t dist = hit_debug ? md_key : inner_prod_plain[i];
                uint64_t idv = id_prime_plain[i];
                uint64_t payload = make_payload_plain(idv, (uint64_t)(i + 1));
                pairs_cand.push_back(std::make_pair(dist, payload));
            }

            std::vector<std::pair<uint64_t, uint64_t>> pairs_c;
            std::vector<std::pair<uint64_t, uint64_t>> pairs_w;
            pairs_c.reserve(total_C);
            pairs_w.reserve(total_W);

            // Verification mirrors the circuit's always-on full-sort path.
            if (true /* was: iter == 0 */) {
                pairs_c = pairs_c_old;
                pairs_w = pairs_w_old;
                pairs_c.insert(pairs_c.end(), pairs_cand.begin(), pairs_cand.end());
                pairs_w.insert(pairs_w.end(), pairs_cand.begin(), pairs_cand.end());
                PlainBitonicFullSortPairs(pairs_c, inner_prod_bitlen);
                PlainBitonicFullSortPairs(pairs_w, inner_prod_bitlen);
            } else {
                // Merge-style verification: sort candidates once, then bitonic-merge into C/W.
                PlainBitonicFullSortPairs(pairs_cand, inner_prod_bitlen);

                uint64_t max_key = (inner_prod_bitlen >= 64)
                    ? ~0ULL
                    : ((1ULL << inner_prod_bitlen) - 1ULL);

                auto pad_pairs = [&](std::vector<std::pair<uint64_t, uint64_t>>& pairs) {
                    uint32_t padded = next_power_of_two((uint32_t)pairs.size());
                    while (pairs.size() < padded) {
                        pairs.push_back(std::make_pair(max_key, 0));
                    }
                };

                auto merge_sorted_pairs =
                    [&](const std::vector<std::pair<uint64_t, uint64_t>>& a,
                        const std::vector<std::pair<uint64_t, uint64_t>>& b,
                        uint32_t keep,
                        std::vector<std::pair<uint64_t, uint64_t>>& out) {
                        out.clear();
                        out.reserve(a.size() + b.size());
                        out.insert(out.end(), a.begin(), a.end());
                        for (size_t i = 0; i < b.size(); i++) {
                            out.push_back(b[b.size() - 1 - i]);
                        }
                        pad_pairs(out);
                        PlainBitonicMergePairs(out);
                        if (out.size() > keep) {
                            out.resize(keep);
                        }
                    };

                merge_sorted_pairs(pairs_c_old, pairs_cand, total_C, pairs_c);
                merge_sorted_pairs(pairs_w_old, pairs_cand, total_W, pairs_w);
            }

            bool exact_c = true;
            bool exact_w = true;
            for (uint32_t i = 0; i < total_C; i++) {
                if (C_key_results[i] != pairs_c[i].first || C_pay_results[i] != pairs_c[i].second) {
                    std::cout << "  C mismatch at " << i << ": got (" << C_key_results[i]
                              << "," << C_pay_results[i] << ") expected ("
                              << pairs_c[i].first << "," << pairs_c[i].second << ")"
                              << std::endl;
                    exact_c = false;
                    break;
                }
            }
            for (uint32_t i = 0; i < total_W; i++) {
                if (W_key_results[i] != pairs_w[i].first || W_pay_results[i] != pairs_w[i].second) {
                    std::cout << "  W mismatch at " << i << ": got (" << W_key_results[i]
                              << "," << W_pay_results[i] << ") expected ("
                              << pairs_w[i].first << "," << pairs_w[i].second << ")"
                              << std::endl;
                    exact_w = false;
                    break;
                }
            }
            if (exact_c && exact_w) {
                std::cout << "  Strong verification passed" << std::endl;
            }
        } else {
            std::cout << "\n  (Strong verification skipped for large scale)" << std::endl;
        }
    }

    delete[] inner_products;
    delete[] C_dist_shares;
    delete[] C_id_shares;
    delete[] W_dist_shares;
    delete[] W_id_shares;
    delete[] C_key_out;
    delete[] C_pay_out;
    delete[] W_key_out;
    delete[] W_pay_out;
    for (share* s : cleanup) {
        delete s;
    }

    return res;
}

int32_t hnsecw_single_b2a(
    e_role role, const std::string& address, uint16_t port,
    seclvl seclvl, uint32_t M, uint32_t D, uint32_t L_C, uint32_t L_W,
    uint32_t LV, uint32_t bitlen, uint32_t id_bitlen,
    uint32_t entry_id, const std::string& entry_file, const std::string& entry_out,
    uint32_t dummy_id_override, uint32_t debug_tag,
    uint32_t nthreads, e_mt_gen_alg mt_alg) {

    uint32_t inner_prod_bitlen = 2 * bitlen + (uint32_t)ceil(log2(D)) + 1;
    uint32_t maxbitlen = inner_prod_bitlen;
    uint32_t arith_bitlen = 0;
    if (maxbitlen <= 8) {
        arith_bitlen = 8;
    } else if (maxbitlen <= 16) {
        arith_bitlen = 16;
    } else if (maxbitlen <= 32) {
        arith_bitlen = 32;
    } else {
        arith_bitlen = 64;
    }

    std::cout << "\n========================================================" << std::endl;
    std::cout << "  Layer Search (T=L_C)" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  M (neighbors): " << M << std::endl;
    std::cout << "  D (dimension): " << D << std::endl;
    std::cout << "  L_C: " << L_C << std::endl;
    std::cout << "  L_W: " << L_W << std::endl;
    std::cout << "  LV: " << LV << std::endl;
    std::cout << "  Input bitlen: " << bitlen << std::endl;
    std::cout << "  ID bitlen: " << id_bitlen << std::endl;
    std::cout << "  Inner product bitlen: " << inner_prod_bitlen << std::endl;
    std::cout << "  Arithmetic share bitlen: " << arith_bitlen << std::endl;
    std::cout << "  Debug tag payload: " << (debug_tag ? "on" : "off") << std::endl;
    std::cout << "========================================================\n" << std::endl;

    if (inner_prod_bitlen > 64) {
        std::cerr << "Inner product bitlen exceeds 64; reduce bitlen or D." << std::endl;
        return 1;
    }

    uint32_t T = L_C;
    uint32_t Vmax = 1 + T * M;
    std::cout << "Vmax (visited length) = " << Vmax << std::endl;

    srand(12345);

    LoadRealDataIfRequested();
    uint32_t U = 1u << id_bitlen;
    uint32_t visited_pad = U - 1;
    uint32_t dummy_pool = T * M;
    uint32_t N = g_real_data.loaded ? g_real_data.N : (U - 1 - dummy_pool);
    assert(N > 0);
    if (g_real_data.loaded && (uint64_t)N + (uint64_t)dummy_pool >= (uint64_t)U) {
        std::cerr << "[HNSECW_DATA_FILE] N=" << N << " + dummy_pool=" << dummy_pool
                  << " exceeds U=" << U << " (id_bitlen=" << id_bitlen
                  << "); increase id_bitlen." << std::endl;
        return 1;
    }

    std::vector<uint32_t> query_plain(D);
    if (g_real_data.loaded && g_real_data.num_queries > 0 && D == g_real_data.D) {
        query_plain = g_real_data.query[0];
    } else {
        for (uint32_t j = 0; j < D; j++) {
            uint32_t bound = (bitlen >= 2 && bitlen < 32) ? (1U << (bitlen - 1)) : (1U << 31);
            query_plain[j] = rand() % bound;
        }
    }

    uint64_t pad_key = (inner_prod_bitlen >= 64) ? ~0ULL : ((1ULL << inner_prod_bitlen) - 1ULL);
    uint64_t md_key = (pad_key == 0) ? 0 : (pad_key - 1);
    uint32_t c_mod = (inner_prod_bitlen >= 32) ? (1U << 31) : (uint32_t)md_key;

    std::vector<uint32_t> C_dist_plain(L_C);
    std::vector<uint32_t> C_id_plain(L_C);
    std::vector<uint32_t> W_dist_plain(L_W);
    std::vector<uint32_t> W_id_plain(L_W);
    // Paper-spec init: (max_dist, dummy_id) sentinels so unfilled slots
    // sort to the bottom of the candidate set.
    uint32_t init_dummy_id = (dummy_id_override != kDummyIdAuto)
                                ? dummy_id_override
                                : N;
    for (uint32_t i = 0; i < L_C; i++) {
        C_dist_plain[i] = (uint32_t)md_key;
        C_id_plain[i] = init_dummy_id;
    }
    for (uint32_t i = 0; i < L_W; i++) {
        W_dist_plain[i] = (uint32_t)md_key;
        W_id_plain[i] = init_dummy_id;
    }

    uint32_t entry = rand() % N;
    if (!entry_file.empty()) {
        entry = LoadEntryFromFile(entry_file);
    } else if (entry_id != kEntryAuto) {
        entry = entry_id;
    }
    uint32_t u = entry;

    std::vector<uint32_t> visited_plain(Vmax, visited_pad);
    std::unordered_set<uint32_t> visited_set;
    visited_plain[0] = entry;
    visited_set.insert(entry);
    uint32_t visited_count = 1;
    uint16_t port_execA = port;
    uint16_t port_execB = (uint16_t)(port + 1);
    double total_online_ms = 0.0;
    uint64_t total_online_bytes = 0;

    // Reuse parties across iterations; circuits are reset inside each Exec.
    // ExecA only needs to represent sum in [0, Vmax], so smaller arithmetic bitlen is enough.
    uint32_t need_bits = ceil_log2_u32(Vmax + 1);
    if (need_bits == 0) {
        need_bits = 1;
    }
    uint32_t arith_bitlen_A = (need_bits <= 8) ? 8
        : (need_bits <= 16) ? 16
        : (need_bits <= 32) ? 32
        : 64;
    // Recreate parties each iteration: ABYParty::Reset() leaves the arith
    // Beaver-triple consumer state shifted, so MULGate outputs from iter ~3
    // mix stale triples and yield distances that exceed the natural bound.
    ABYParty* partyA = nullptr;
    ABYParty* partyB = nullptr;

    for (uint32_t t = 0; t < T; t++) {
        // Per-iter recreate (see hnsecw_single_b2y.cpp for rationale).
        delete partyA;
        delete partyB;
        partyA = new ABYParty(role, address, port_execA, seclvl,
                              arith_bitlen_A, nthreads, mt_alg);
        partyB = new ABYParty(role, address, port_execB, seclvl,
                              arith_bitlen, nthreads, mt_alg);
        std::cout << "\n================ Iteration " << t << " ================" << std::endl;

        std::vector<uint32_t> neigh_ids = GenerateNeighbors(u, t, M, N, u, dummy_id_override);

        std::vector<uint32_t> dummy_id_plain(M);
        for (uint32_t k = 0; k < M; k++) {
            if (dummy_id_override == kDummyIdAuto) {
                dummy_id_plain[k] = N + (t * M + k);
            } else {
                dummy_id_plain[k] = dummy_id_override;
            }
        }

        uint32_t visited_len_eff = visited_count;
        ExecAResult execA;
        bool fast_execA = false;
        if (t == 0 && visited_len_eff == 1 && visited_plain[0] == u) {
            bool has_self = false;
            for (uint32_t idv : neigh_ids) {
                if (idv == u) {
                    has_self = true;
                    break;
                }
            }
            if (!has_self) {
                fast_execA = true;
                execA.id_prime = neigh_ids;
                execA.hit_share.assign(M, 0);
                execA.online_ms = 0.0;
                execA.online_bytes = 0;
                if (debug_tag && role == SERVER) {
                    std::cout << "[ExecA t=0] Fast-path: skip visited check (no EP in neighbors)" << std::endl;
                }
            }
        }
        if (!fast_execA) {
            execA = RunExecA(partyA, role,
                             M, id_bitlen,
                             neigh_ids, dummy_id_plain, visited_plain, visited_len_eff,
                             debug_tag,
                             "[ExecA t=" + std::to_string(t) + "]");
        }

        std::vector<std::vector<uint32_t>> fetched_vectors =
            FetchVectors(execA.id_prime, N, D, bitlen, dummy_id_override);

        ExecBResult execB = RunExecB(partyB, role,
                                     M, D, L_C, L_W, bitlen, id_bitlen,
                                     inner_prod_bitlen, debug_tag, t,
                                     query_plain, fetched_vectors,
                                     execA.hit_share, execA.id_prime, N,
                                     C_dist_plain, C_id_plain,
                                     W_dist_plain, W_id_plain, md_key,
                                     "[ExecB t=" + std::to_string(t) + "]");
        total_online_ms += execA.online_ms + execB.online_ms;
        total_online_bytes += execA.online_bytes + execB.online_bytes;

        C_dist_plain = execB.C_dist;
        C_id_plain = execB.C_id;
        W_dist_plain = execB.W_dist;
        W_id_plain = execB.W_id;

        for (uint32_t k = 0; k < M; k++) {
            uint32_t idv = execA.id_prime[k];
            assert(visited_set.find(idv) == visited_set.end());
            visited_plain[visited_count++] = idv;
            visited_set.insert(idv);
        }
        assert(visited_count == 1 + (t + 1) * M);

        if (!C_id_plain.empty()) {
            u = C_id_plain[0];
            // Pop the just-used u_next so the next iteration's bitonic
            // merge sorts this slot to the bottom; without the pop the
            // running global min stays as C[0] forever and the search
            // stalls at one node.  Standard HNSW (Malkov 2018) extracts
            // the closest from C at each step.
            C_dist_plain[0] = (uint32_t)md_key;
            C_id_plain[0] = init_dummy_id;
        }
        std::cout << "[State] u_next=" << u << " visited_count=" << visited_count << std::endl;
    }

    if (role == CLIENT && !entry_out.empty() && !W_id_plain.empty()) {
        // Walk W and pick the first id within [0, N_real).  Same fallback
        // as batch_b2y / multi_b2y_dyn / single_b2y.
        uint32_t out_id = u;
        for (size_t i = 0; i < W_id_plain.size(); ++i) {
            if (g_real_data.loaded ? (W_id_plain[i] < g_real_data.N) : (W_id_plain[i] < N)) {
                out_id = W_id_plain[i];
                break;
            }
        }
        WriteEntryToFile(entry_out, out_id);
    }

    std::cout << "\n[Total Online] latency(s)=" << (total_online_ms / 1000.0)
              << " comm(MB)=" << (total_online_bytes / (1024.0 * 1024.0))
              << std::endl;
    delete partyB;
    delete partyA;
    return 0;
}
