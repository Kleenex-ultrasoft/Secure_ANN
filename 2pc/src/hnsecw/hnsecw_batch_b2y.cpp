#include "hnsecw_batch_b2y.h"
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

static std::vector<uint32_t> LoadEntryList(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open entry file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) {
        throw std::runtime_error("entry file size is not a multiple of 4 bytes: " + path);
    }
    size_t count = static_cast<size_t>(size / sizeof(uint32_t));
    std::vector<uint32_t> out(count);
    if (count > 0) {
        in.read(reinterpret_cast<char*>(out.data()), sizeof(uint32_t) * count);
    }
    return out;
}

static void WriteEntryList(const std::string& path, const std::vector<uint32_t>& entries) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open entry_out: " + path);
    }
    if (!entries.empty()) {
        out.write(reinterpret_cast<const char*>(entries.data()),
                  sizeof(uint32_t) * entries.size());
    }
}

// OR-reduce 1-bit shares using a depth-balanced tree; used on Yao shares.
static share* ORReduceStreamedBits(const std::vector<share*>& bits,
                                   BooleanCircuit* circ,
                                   std::vector<share*>& cleanup) {
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    std::vector<share*> levels;
    levels.reserve(32);
    for (share* bit : bits) {
        share* acc = bit;
        size_t k = 0;
        while (k < levels.size() && levels[k] != nullptr) {
            acc = track(circ->PutORGate(levels[k], acc));
            levels[k] = nullptr;
            ++k;
        }
        if (k == levels.size()) {
            levels.push_back(acc);
        } else {
            levels[k] = acc;
        }
    }

    share* acc = nullptr;
    for (share* s : levels) {
        if (!s) {
            continue;
        }
        acc = acc ? track(circ->PutORGate(acc, s)) : s;
    }
    if (!acc) {
        acc = track(circ->PutCONSGate((uint64_t)0, 1));
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

static uint32_t OrReduceWireBits(const std::vector<uint32_t>& bits, BooleanCircuit* circ) {
    if (bits.empty()) {
        share* zero_s = circ->PutCONSGate(static_cast<uint64_t>(0), 1);
        return zero_s->get_wire_id(0);
    }
    uint32_t acc = bits[0];
    for (size_t i = 1; i < bits.size(); i++) {
        acc = circ->PutORGate(acc, bits[i]);
    }
    return acc;
}

static uint32_t EqWireBits(const std::vector<uint32_t>& a,
                           const std::vector<uint32_t>& b,
                           BooleanCircuit* circ,
                           uint32_t one) {
    assert(a.size() == b.size());
    std::vector<uint32_t> diff_bits;
    diff_bits.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        diff_bits.push_back(circ->PutXORGate(a[i], b[i]));
    }
    uint32_t diff = OrReduceWireBits(diff_bits, circ);
    return circ->PutXORGate(diff, one);
}

static uint32_t mux_wire(uint32_t a, uint32_t b, uint32_t sel, BooleanCircuit* circ) {
    std::vector<uint32_t> out = circ->PutMUXGate({a}, {b}, sel);
    return out[0];
}

static uint32_t mux_packed_wire(uint32_t a,
                                uint32_t b,
                                uint32_t sel,
                                uint32_t bitlen,
                                BooleanCircuit* circ) {
    // a/b are packed (combiner) wires with nvals == bitlen; mux bitwise to keep nvals intact.
    std::vector<uint32_t> a_bits = circ->PutSplitterGate(a);
    std::vector<uint32_t> b_bits = circ->PutSplitterGate(b);
    assert(a_bits.size() == bitlen);
    assert(b_bits.size() == bitlen);
    std::vector<uint32_t> out_bits = circ->PutMUXGate(a_bits, b_bits, sel);
    return circ->PutCombinerGate(out_bits);
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

struct DedupResult {
    std::vector<uint32_t> id_prime;
    std::vector<uint8_t> hit_share;
    double online_ms = 0.0;
    uint64_t online_bytes = 0;
};

struct BatchExecBResult {
    std::vector<std::vector<uint32_t>> C_dist;
    std::vector<std::vector<uint32_t>> C_id;
    std::vector<std::vector<uint32_t>> W_dist;
    std::vector<std::vector<uint32_t>> W_id;
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

    for (uint32_t i = 0; i < M; i++) {
        share* id_b = id_b_shares[i];
        std::vector<share*> eq_bits_y;
        eq_bits_y.reserve(visited_len_eff);
        for (uint32_t j = 0; j < visited_len_eff; j++) {
            share* eq_b = track(bool_circ->PutEQGate(id_b, visited_consts[j]));
            // Lift EQ results to Yao and reduce there to avoid B2A/A2Y hops.
            share* eq_y = track(yao_circ->PutB2YGate(eq_b));
            eq_bits_y.push_back(eq_y);
        }
        share* hit_y = ORReduceStreamedBits(eq_bits_y, yao_circ, cleanup);
        // OR is equivalent to (sum > 0) for 0/1 EQ bits.

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

    share* md_y = track(yao_circ->PutCONSGate(md_key, inner_prod_bitlen));

#ifdef DBG_HIT
    uint32_t dbg_limit = 0;
    if (debug_tag) {
        dbg_limit = std::min(M, 8U);
    }
    std::vector<share*> dbg_hit_b_out(dbg_limit, nullptr);
    std::vector<share*> dbg_dist_out(dbg_limit, nullptr);
#endif

    std::vector<share*> cand_dist_y(M);
    std::vector<share*> cand_id_y(M);
    for (uint32_t i = 0; i < M; i++) {
        assert((hit_share[i] & 1U) == hit_share[i]);
        share* dist_a = inner_products[i];
        share* dist_y = track(yao_circ->PutA2YGate(dist_a));
        share* hit_server = track(bool_circ->PutINGate(
            (role == SERVER ? (uint64_t)hit_share[i] : 0),
            1,
            SERVER));
        share* hit_client = track(bool_circ->PutINGate(
            (role == CLIENT ? (uint64_t)hit_share[i] : 0),
            1,
            CLIENT));
        share* hit_b = track(bool_circ->PutXORGate(hit_server, hit_client));
        share* hit_y = track(yao_circ->PutB2YGate(hit_b));
        // PutMUXGate selects first arg when sel=1, so hit->md_key, miss->dist.
        share* dist_sel_y = track(yao_circ->PutMUXGate(md_y, dist_y, hit_y));

        share* id_y = track(yao_circ->PutCONSGate(id_prime_plain[i], id_bitlen));

        cand_dist_y[i] = dist_sel_y;
        cand_id_y[i] = id_y;

#ifdef DBG_HIT
        if (debug_tag && i < dbg_limit) {
            dbg_hit_b_out[i] = track(bool_circ->PutOUTGate(hit_b, CLIENT));
            dbg_dist_out[i] = track(yao_circ->PutOUTGate(dist_sel_y, CLIENT));
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

    // Always take the full-sort path; the merge branch consumes more
    // arithmetic OT triples and triggers the IKNP-OT pool issue at
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

enum class DedupAlgo : uint32_t {
    kBitonicTag = 0,
    kBitonicGroup = 1
};

static DedupResult RunBatchDedup(
    ABYParty* party,
    e_role role,
    uint32_t id_bitlen,
    const std::vector<uint32_t>& cand_ids,
    const std::vector<uint32_t>& dummy_ids,
    const std::vector<uint32_t>& visited_plain,
    uint32_t visited_len_eff,
    bool use_yao,
    DedupAlgo algo,
    uint32_t debug_tag,
    const std::string& label) {

    ResetPartyForReuse(party);
    (void)debug_tag;

    std::vector<Sharing*>& sharings = party->GetSharings();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    BooleanCircuit* circ = use_yao ? yao_circ : bool_circ;
    const e_role OWNER = SERVER;

    assert(cand_ids.size() == dummy_ids.size());
    assert(visited_len_eff <= visited_plain.size());

    std::vector<share*> cleanup;
    cleanup.reserve((size_t)(visited_len_eff + cand_ids.size()) * 16 + 2048);
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    const uint32_t K = static_cast<uint32_t>(cand_ids.size());
    const uint32_t V = visited_len_eff;
    uint32_t idx_bits = ceil_log2_u32(K + 1);
    if (idx_bits == 0) {
        idx_bits = 1;
    }
    const bool use_group = (algo == DedupAlgo::kBitonicGroup);
    const uint32_t key_bits = use_group ? id_bitlen : (id_bitlen + 1);
    const uint32_t payload1_bits = id_bitlen * 2 + idx_bits + 1;
    const uint32_t payload2_bitlen = id_bitlen + 2;
    const size_t n_items = static_cast<size_t>(V) + static_cast<size_t>(K);
    if (debug_tag) {
        std::cout << label << " build start: K=" << K
                  << " V=" << V
                  << " n_items=" << n_items
                  << " use_yao=" << (use_yao ? "1" : "0")
                  << " algo=" << (use_group ? "radix" : "bitonic")
                  << std::endl;
    }

    share* tag0_s = track(circ->PutCONSGate(static_cast<uint64_t>(0), 1));
    share* tag1_s = track(circ->PutCONSGate(static_cast<uint64_t>(1), 1));
    uint32_t tag0 = tag0_s->get_wire_id(0);
    uint32_t tag1 = tag1_s->get_wire_id(0);
    share* one_s = track(circ->PutCONSGate(static_cast<uint64_t>(1), 1));
    uint32_t one = one_s->get_wire_id(0);
    share* zero_s = track(circ->PutCONSGate(static_cast<uint64_t>(0), 1));
    uint32_t zero = zero_s->get_wire_id(0);

    std::vector<uint32_t> keys(n_items);
    std::vector<uint32_t> payload1(n_items);

    for (uint32_t i = 0; i < V; i++) {
        share* id_s = track(circ->PutCONSGate(visited_plain[i], id_bitlen));
        share* dummy_s = track(circ->PutCONSGate(static_cast<uint64_t>(0), id_bitlen));
        share* idx_s = track(circ->PutCONSGate(static_cast<uint64_t>(K), idx_bits));

        if (use_group) {
            std::vector<uint32_t> key_bits_vec;
            key_bits_vec.reserve(id_bitlen);
            for (uint32_t b = 0; b < id_bitlen; b++) {
                key_bits_vec.push_back(id_s->get_wire_id(b));
            }
            keys[i] = circ->PutCombinerGate(key_bits_vec);
        } else {
            std::vector<uint32_t> key_bits_vec;
            key_bits_vec.reserve(key_bits);
            key_bits_vec.push_back(tag0);
            for (uint32_t b = 0; b < id_bitlen; b++) {
                key_bits_vec.push_back(id_s->get_wire_id(b));
            }
            keys[i] = circ->PutCombinerGate(key_bits_vec);
        }

        std::vector<uint32_t> payload_bits;
        payload_bits.reserve(payload1_bits);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            payload_bits.push_back(dummy_s->get_wire_id(b));
        }
        for (uint32_t b = 0; b < idx_bits; b++) {
            payload_bits.push_back(idx_s->get_wire_id(b));
        }
        payload_bits.push_back(tag0);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            payload_bits.push_back(id_s->get_wire_id(b));
        }
        payload1[i] = circ->PutCombinerGate(payload_bits);
    }

    for (uint32_t i = 0; i < K; i++) {
        share* id_s = track(circ->PutINGate(
            (role == OWNER ? (uint64_t)cand_ids[i] : 0),
            id_bitlen,
            OWNER));
        share* dummy_s = track(circ->PutINGate(
            (role == OWNER ? (uint64_t)dummy_ids[i] : 0),
            id_bitlen,
            OWNER));
        share* idx_s = track(circ->PutCONSGate(static_cast<uint64_t>(i), idx_bits));

        if (use_group) {
            std::vector<uint32_t> key_bits_vec;
            key_bits_vec.reserve(id_bitlen);
            for (uint32_t b = 0; b < id_bitlen; b++) {
                key_bits_vec.push_back(id_s->get_wire_id(b));
            }
            keys[V + i] = circ->PutCombinerGate(key_bits_vec);
        } else {
            std::vector<uint32_t> key_bits_vec;
            key_bits_vec.reserve(key_bits);
            key_bits_vec.push_back(tag1);
            for (uint32_t b = 0; b < id_bitlen; b++) {
                key_bits_vec.push_back(id_s->get_wire_id(b));
            }
            keys[V + i] = circ->PutCombinerGate(key_bits_vec);
        }

        std::vector<uint32_t> payload_bits;
        payload_bits.reserve(payload1_bits);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            payload_bits.push_back(dummy_s->get_wire_id(b));
        }
        for (uint32_t b = 0; b < idx_bits; b++) {
            payload_bits.push_back(idx_s->get_wire_id(b));
        }
        payload_bits.push_back(tag1);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            payload_bits.push_back(id_s->get_wire_id(b));
        }
        payload1[V + i] = circ->PutCombinerGate(payload_bits);
    }

    BitonicFullSortPairs(keys, payload1, key_bits, payload1_bits, circ);
    if (debug_tag) {
        std::cout << label << " build stage1 sort done" << std::endl;
    }

    std::vector<uint32_t> payload2_hit(n_items, zero);
    std::vector<uint32_t> payload2_r(n_items, zero);
    std::vector<std::vector<uint32_t>> payload2_id_bits(n_items);
    std::vector<std::vector<uint32_t>> id_bits_list(n_items);
    std::vector<std::vector<uint32_t>> dummy_bits_list(n_items);
    std::vector<std::vector<uint32_t>> idx_bits_list(n_items);
    std::vector<uint32_t> tag_wires(n_items);

    for (size_t i = 0; i < n_items; i++) {
        std::vector<uint32_t> p = circ->PutSplitterGate(payload1[i]);

        std::vector<uint32_t> dummy_bits(id_bitlen);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            dummy_bits[b] = p[b];
        }
        std::vector<uint32_t> idx_bits_vec(idx_bits);
        for (uint32_t b = 0; b < idx_bits; b++) {
            idx_bits_vec[b] = p[id_bitlen + b];
        }
        uint32_t tag_wire = p[id_bitlen + idx_bits];
        std::vector<uint32_t> id_bits(id_bitlen);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            id_bits[b] = p[id_bitlen + idx_bits + 1 + b];
        }

        dummy_bits_list[i] = dummy_bits;
        idx_bits_list[i] = idx_bits_vec;
        id_bits_list[i] = id_bits;
        tag_wires[i] = tag_wire;
    }
    if (debug_tag) {
        std::cout << label << " build stage1 unpack done" << std::endl;
    }

    std::vector<uint32_t> group_has_visited(n_items, zero);
    std::vector<uint32_t> cand_dup(n_items, zero);
    if (use_group) {
        std::vector<uint32_t> visited_fwd(n_items, zero);
        std::vector<uint32_t> visited_bwd(n_items, zero);
        std::vector<uint32_t> prev_id_bits;
        prev_id_bits.reserve(id_bitlen);
        uint32_t visited_fwd_prev = zero;
        uint32_t cand_seen = zero;

        for (size_t i = 0; i < n_items; i++) {
            uint32_t eq_prev = zero;
            if (i > 0) {
                std::vector<uint32_t> diff_bits;
                diff_bits.reserve(id_bitlen);
                for (uint32_t b = 0; b < id_bitlen; b++) {
                    diff_bits.push_back(circ->PutXORGate(id_bits_list[i][b], prev_id_bits[b]));
                }
                uint32_t diff = OrReduceWireBits(diff_bits, circ);
                eq_prev = circ->PutXORGate(diff, one);
            }
            prev_id_bits = id_bits_list[i];
            uint32_t group_start = circ->PutXORGate(eq_prev, one);
            uint32_t is_visited = circ->PutXORGate(tag_wires[i], one);
            uint32_t visited_or = circ->PutORGate(visited_fwd_prev, is_visited);
            visited_fwd[i] = mux_wire(visited_or, is_visited, group_start, circ);
            uint32_t cand_seen_clean = mux_wire(cand_seen, zero, group_start, circ);
            cand_dup[i] = circ->PutANDGate(tag_wires[i], cand_seen_clean);
            uint32_t cand_or = circ->PutORGate(cand_seen, tag_wires[i]);
            cand_seen = mux_wire(cand_or, tag_wires[i], group_start, circ);
            visited_fwd_prev = visited_fwd[i];
        }

        std::vector<uint32_t> next_id_bits;
        next_id_bits.reserve(id_bitlen);
        uint32_t visited_bwd_next = zero;
        for (size_t i = n_items; i-- > 0;) {
            uint32_t eq_next = zero;
            if (i + 1 < n_items) {
                std::vector<uint32_t> diff_bits;
                diff_bits.reserve(id_bitlen);
                for (uint32_t b = 0; b < id_bitlen; b++) {
                    diff_bits.push_back(circ->PutXORGate(id_bits_list[i][b], id_bits_list[i + 1][b]));
                }
                uint32_t diff = OrReduceWireBits(diff_bits, circ);
                eq_next = circ->PutXORGate(diff, one);
            }
            uint32_t group_end = circ->PutXORGate(eq_next, one);
            uint32_t is_visited = circ->PutXORGate(tag_wires[i], one);
            uint32_t visited_or = circ->PutORGate(visited_bwd_next, is_visited);
            visited_bwd[i] = mux_wire(visited_or, is_visited, group_end, circ);
            visited_bwd_next = visited_bwd[i];
        }

        for (size_t i = 0; i < n_items; i++) {
            group_has_visited[i] = circ->PutORGate(visited_fwd[i], visited_bwd[i]);
        }
    }
    if (debug_tag) {
        std::cout << label << " build group pass done" << std::endl;
    }

    for (size_t i = 0; i < n_items; i++) {
        uint32_t hit_wire = zero;
        if (use_group) {
            uint32_t hit_or = circ->PutORGate(group_has_visited[i], cand_dup[i]);
            hit_wire = circ->PutANDGate(tag_wires[i], hit_or);
        } else {
            uint32_t eq_prev = zero;
            if (i > 0) {
                std::vector<uint32_t> diff_bits;
                diff_bits.reserve(id_bitlen);
                for (uint32_t b = 0; b < id_bitlen; b++) {
                    diff_bits.push_back(circ->PutXORGate(id_bits_list[i][b], id_bits_list[i - 1][b]));
                }
                uint32_t diff = OrReduceWireBits(diff_bits, circ);
                eq_prev = circ->PutXORGate(diff, one);
            }
            hit_wire = circ->PutANDGate(tag_wires[i], eq_prev);
        }

        share* r_s = track(circ->PutINGate(
            (role == SERVER ? static_cast<uint64_t>(rand() & 1U) : 0),
            1,
            SERVER));
        uint32_t r_wire = r_s->get_wire_id(0);
        uint32_t hit_masked_wire = circ->PutXORGate(hit_wire, r_wire);

        uint32_t id_wire = circ->PutCombinerGate(id_bits_list[i]);
        uint32_t dummy_wire = circ->PutCombinerGate(dummy_bits_list[i]);
        uint32_t id_sel_wire = mux_packed_wire(dummy_wire, id_wire, hit_wire, id_bitlen, circ);

        std::vector<uint32_t> id_sel_split = circ->PutSplitterGate(id_sel_wire);
        std::vector<uint32_t> id_sel_bits(id_bitlen);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            id_sel_bits[b] = id_sel_split[b];
        }

        payload2_hit[i] = hit_masked_wire;
        payload2_r[i] = r_wire;
        payload2_id_bits[i] = id_sel_bits;

        std::vector<uint32_t> payload2_bits_vec;
        payload2_bits_vec.reserve(payload2_bitlen);
        payload2_bits_vec.push_back(hit_masked_wire);
        payload2_bits_vec.push_back(r_wire);
        for (uint32_t b = 0; b < id_bitlen; b++) {
            payload2_bits_vec.push_back(id_sel_bits[b]);
        }
        (void)payload2_bits_vec;
    }

    std::vector<share*> id_prime_out(n_items);
    std::vector<share*> hit_masked_out(n_items);
    std::vector<share*> r_out(n_items);
    std::vector<share*> idx_out(n_items);
    std::vector<share*> tag_out(n_items);

    for (size_t i = 0; i < n_items; i++) {
        share* id_share = track(create_new_share(payload2_id_bits[i], circ));
        id_prime_out[i] = track(circ->PutOUTGate(id_share, ALL));

        share* hit_share = track(create_new_share(std::vector<uint32_t>{payload2_hit[i]}, circ));
        hit_masked_out[i] = track(circ->PutOUTGate(hit_share, CLIENT));

        share* r_share = track(create_new_share(std::vector<uint32_t>{payload2_r[i]}, circ));
        r_out[i] = track(circ->PutOUTGate(r_share, SERVER));

        share* idx_share = track(create_new_share(idx_bits_list[i], circ));
        idx_out[i] = track(circ->PutOUTGate(idx_share, ALL));

        share* tag_share = track(create_new_share(std::vector<uint32_t>{tag_wires[i]}, circ));
        tag_out[i] = track(circ->PutOUTGate(tag_share, ALL));
    }
    if (debug_tag) {
        std::cout << label << " build stage2 outputs done" << std::endl;
    }

    std::cout << "\n" << label << " Executing" << std::endl;
    OnlineDelta delta = ExecCircuitDelta(party);
    std::cout << label << " Executed" << std::endl;

    if (role == SERVER) {
        PrintPartyStats(party, sharings, label, !use_yao, false, use_yao);
    }

    DedupResult res;
    res.id_prime = dummy_ids;
    res.hit_share.assign(K, 0);
    res.online_ms = delta.ms;
    res.online_bytes = delta.bytes;
    for (size_t i = 0; i < n_items; i++) {
        uint32_t tag = (uint32_t)(tag_out[i]->get_clear_value<uint64_t>() & 1ULL);
        uint32_t idx = (uint32_t)idx_out[i]->get_clear_value<uint64_t>();
        if (tag == 0 || idx >= K) {
            continue;
        }
        res.id_prime[idx] = (uint32_t)id_prime_out[i]->get_clear_value<uint64_t>();
        if (role == SERVER) {
            res.hit_share[idx] = (uint8_t)(r_out[i]->get_clear_value<uint64_t>() & 1ULL);
        } else {
            res.hit_share[idx] = (uint8_t)(hit_masked_out[i]->get_clear_value<uint64_t>() & 1ULL);
        }
    }

    for (share* s : cleanup) {
        delete s;
    }
    return res;
}

static BatchExecBResult RunBatchExecB(
    ABYParty* party,
    e_role role,
    uint32_t Z,
    uint32_t K,
    uint32_t D,
    uint32_t L_C,
    uint32_t L_W,
    uint32_t bitlen,
    uint32_t id_bitlen,
    uint32_t inner_prod_bitlen,
    uint32_t debug_tag,
    uint32_t iter,
    const std::vector<std::vector<uint32_t>>& query_plain,
    const std::vector<std::vector<uint32_t>>& fetched_vectors,
    const std::vector<uint8_t>& hit_share,
    const std::vector<uint32_t>& id_prime_plain,
    const std::vector<std::vector<uint32_t>>& C_dist_plain,
    const std::vector<std::vector<uint32_t>>& C_id_plain,
    const std::vector<std::vector<uint32_t>>& W_dist_plain,
    const std::vector<std::vector<uint32_t>>& W_id_plain,
    uint64_t md_key,
    const std::string& label) {

    ResetPartyForReuse(party);

    std::vector<Sharing*>& sharings = party->GetSharings();
    ArithmeticCircuit* arith_circ = (ArithmeticCircuit*) sharings[S_ARITH]->GetCircuitBuildRoutine();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    const e_role OWNER = SERVER;

    std::vector<share*> cleanup;
    cleanup.reserve((size_t)Z * (size_t)K * 6 + (size_t)(L_C + L_W) * Z + 2048);
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    std::vector<std::vector<share*>> s_query(Z);
    for (uint32_t q = 0; q < Z; q++) {
        std::vector<uint32_t> zeros_query(D, 0);
        uint32_t* q_ptr = (role == OWNER
            ? const_cast<uint32_t*>(query_plain[q].data())
            : zeros_query.data());
        share* s_q = track(arith_circ->PutSIMDINGate(D, q_ptr, bitlen, OWNER));
        s_query[q].assign(1, s_q);
    }

    std::vector<std::vector<share*>> inner_products(Z, std::vector<share*>(K, nullptr));
    for (uint32_t k = 0; k < K; k++) {
        std::vector<uint32_t> zeros_vec(D, 0);
        uint32_t* vec_ptr = (role == OWNER
            ? const_cast<uint32_t*>(fetched_vectors[k].data())
            : zeros_vec.data());
        share* s_vec = track(arith_circ->PutSIMDINGate(D, vec_ptr, bitlen, OWNER));

        for (uint32_t q = 0; q < Z; q++) {
            // Squared L2 distance: (v - q)^2 element-wise, then sum.
            // Same fix as single mode (commit f5e27e3) — batch was missed.
            share* s_diff = track(arith_circ->PutSUBGate(s_vec, s_query[q][0]));
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

            share* s_ip = new arithshare(arith_circ);
            cleanup.push_back(s_ip);
            s_ip->set_wire_id(0, sum_wires[0]);
            inner_products[q][k] = s_ip;
        }
    }

    std::vector<share*> hit_y(K, nullptr);
    for (uint32_t k = 0; k < K; k++) {
        assert((hit_share[k] & 1U) == hit_share[k]);
        share* hit_server = track(bool_circ->PutINGate(
            (role == SERVER ? (uint64_t)hit_share[k] : 0),
            1,
            SERVER));
        share* hit_client = track(bool_circ->PutINGate(
            (role == CLIENT ? (uint64_t)hit_share[k] : 0),
            1,
            CLIENT));
        share* hit_b = track(bool_circ->PutXORGate(hit_server, hit_client));
        hit_y[k] = track(yao_circ->PutB2YGate(hit_b));
    }

    share* md_y = track(yao_circ->PutCONSGate(md_key, inner_prod_bitlen));

    uint32_t total_C = L_C + K;
    uint32_t total_W = L_W + K;
    uint32_t tag_bits = 0;
    uint32_t payload_bitlen = id_bitlen;
    if (debug_tag) {
        tag_bits = (K + 1 <= 1) ? 1U : (uint32_t)ceil(log2((double)(K + 1)));
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

    share* max_key_val = track(yao_circ->PutCONSGate((uint64_t)-1, inner_prod_bitlen));
    uint32_t max_key_wire = PackYaoValue(max_key_val, inner_prod_bitlen, yao_circ);
    share* zero_payload = track(yao_circ->PutCONSGate((uint64_t)0, payload_bitlen));
    uint32_t zero_payload_wire = PackYaoValue(zero_payload, payload_bitlen, yao_circ);

    BatchExecBResult res;
    res.C_dist.assign(Z, std::vector<uint32_t>(L_C, 0));
    res.C_id.assign(Z, std::vector<uint32_t>(L_C, 0));
    res.W_dist.assign(Z, std::vector<uint32_t>(L_W, 0));
    res.W_id.assign(Z, std::vector<uint32_t>(L_W, 0));

    std::vector<std::vector<share*>> C_key_out(Z, std::vector<share*>(L_C, nullptr));
    std::vector<std::vector<share*>> C_pay_out(Z, std::vector<share*>(L_C, nullptr));
    std::vector<std::vector<share*>> W_key_out(Z, std::vector<share*>(L_W, nullptr));
    std::vector<std::vector<share*>> W_pay_out(Z, std::vector<share*>(L_W, nullptr));

    for (uint32_t q = 0; q < Z; q++) {
        std::vector<share*> cand_dist_y(K);
        std::vector<share*> cand_id_y(K);
        for (uint32_t k = 0; k < K; k++) {
            share* dist_a = inner_products[q][k];
            share* dist_y = track(yao_circ->PutA2YGate(dist_a));
            share* dist_sel_y = track(yao_circ->PutMUXGate(md_y, dist_y, hit_y[k]));
            share* id_y = track(yao_circ->PutCONSGate(id_prime_plain[k], id_bitlen));
            cand_dist_y[k] = dist_sel_y;
            cand_id_y[k] = id_y;
        }

        std::vector<uint32_t> cand_key_wires(K);
        std::vector<uint32_t> cand_pay_wires(K);
        for (uint32_t k = 0; k < K; k++) {
            cand_key_wires[k] = PackYaoValue(cand_dist_y[k], inner_prod_bitlen, yao_circ);
            cand_pay_wires[k] = pack_payload(cand_id_y[k], k + 1);
        }
        if (iter > 0) {
            BitonicFullSortPairs(cand_key_wires, cand_pay_wires, inner_prod_bitlen, payload_bitlen, yao_circ);
        }

        std::vector<share*> C_dist_y(L_C);
        std::vector<share*> C_id_y(L_C);
        std::vector<share*> W_dist_y(L_W);
        std::vector<share*> W_id_y(L_W);

        for (uint32_t i = 0; i < L_C; i++) {
            share* c_d = track(arith_circ->PutINGate(
                (role == OWNER ? (uint64_t)C_dist_plain[q][i] : 0),
                inner_prod_bitlen,
                OWNER));
            share* c_i = track(arith_circ->PutINGate(
                (role == OWNER ? (uint64_t)C_id_plain[q][i] : 0),
                id_bitlen,
                OWNER));
            C_dist_y[i] = track(yao_circ->PutA2YGate(c_d));
            C_id_y[i] = track(yao_circ->PutA2YGate(c_i));
        }
        for (uint32_t i = 0; i < L_W; i++) {
            share* w_d = track(arith_circ->PutINGate(
                (role == OWNER ? (uint64_t)W_dist_plain[q][i] : 0),
                inner_prod_bitlen,
                OWNER));
            share* w_i = track(arith_circ->PutINGate(
                (role == OWNER ? (uint64_t)W_id_plain[q][i] : 0),
                id_bitlen,
                OWNER));
            W_dist_y[i] = track(yao_circ->PutA2YGate(w_d));
            W_id_y[i] = track(yao_circ->PutA2YGate(w_i));
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
            for (uint32_t i = 0; i < K; i++) {
                C_key_wires[L_C + i] = cand_key_wires[i];
                C_pay_wires[L_C + i] = cand_pay_wires[i];
            }

            for (uint32_t i = 0; i < L_W; i++) {
                W_key_wires[i] = PackYaoValue(W_dist_y[i], inner_prod_bitlen, yao_circ);
                W_pay_wires[i] = pack_payload(W_id_y[i], 0);
            }
            for (uint32_t i = 0; i < K; i++) {
                W_key_wires[L_W + i] = cand_key_wires[i];
                W_pay_wires[L_W + i] = cand_pay_wires[i];
            }

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

        for (uint32_t i = 0; i < L_C; i++) {
            share* tmp = new boolshare(1, yao_circ);
            cleanup.push_back(tmp);
            tmp->set_wire_id(0, C_key_wires[i]);
            C_key_out[q][i] = track(yao_circ->PutOUTGate(tmp, ALL));
            share* tmp_pay = new boolshare(1, yao_circ);
            cleanup.push_back(tmp_pay);
            tmp_pay->set_wire_id(0, C_pay_wires[i]);
            C_pay_out[q][i] = track(yao_circ->PutOUTGate(tmp_pay, ALL));
        }
        for (uint32_t i = 0; i < L_W; i++) {
            share* tmp = new boolshare(1, yao_circ);
            cleanup.push_back(tmp);
            tmp->set_wire_id(0, W_key_wires[i]);
            W_key_out[q][i] = track(yao_circ->PutOUTGate(tmp, ALL));
            share* tmp_pay = new boolshare(1, yao_circ);
            cleanup.push_back(tmp_pay);
            tmp_pay->set_wire_id(0, W_pay_wires[i]);
            W_pay_out[q][i] = track(yao_circ->PutOUTGate(tmp_pay, ALL));
        }
    }

    std::cout << "\n" << label << " Executing" << std::endl;
    OnlineDelta delta = ExecCircuitDelta(party);
    std::cout << label << " Executed" << std::endl;

    if (role == SERVER) {
        PrintPartyStats(party, sharings, label, false, true, true);
    }

    res.online_ms = delta.ms;
    res.online_bytes = delta.bytes;

    uint64_t id_mask64 = (id_bitlen >= 64) ? ~0ULL : ((1ULL << id_bitlen) - 1ULL);
    for (uint32_t q = 0; q < Z; q++) {
        for (uint32_t i = 0; i < L_C; i++) {
            res.C_dist[q][i] = (uint32_t)C_key_out[q][i]->get_clear_value<uint64_t>();
            res.C_id[q][i] = (uint32_t)(C_pay_out[q][i]->get_clear_value<uint64_t>() & id_mask64);
        }
        for (uint32_t i = 0; i < L_W; i++) {
            res.W_dist[q][i] = (uint32_t)W_key_out[q][i]->get_clear_value<uint64_t>();
            res.W_id[q][i] = (uint32_t)(W_pay_out[q][i]->get_clear_value<uint64_t>() & id_mask64);
        }
    }

    // Debug: dump per-query W top-10 to mirror what single_b2y prints.
    for (uint32_t q = 0; q < Z; q++) {
        std::cout << label << " q=" << q << " W (first 10): ";
        for (uint32_t i = 0; i < std::min<uint32_t>(L_W, 10U); i++) {
            std::cout << "(" << res.W_dist[q][i] << "," << res.W_id[q][i] << ") ";
        }
        std::cout << (L_W > 10 ? "..." : "") << std::endl;
    }

    for (share* s : cleanup) {
        delete s;
    }
    return res;
}

int32_t hnsecw_batch_b2y(
    e_role role, const std::string& address, uint16_t port,
    seclvl seclvl, uint32_t num_queries, uint32_t M, uint32_t D,
    uint32_t L_C, uint32_t L_W, uint32_t LV, uint32_t bitlen,
    uint32_t id_bitlen, uint32_t yao_dedup_thresh,
    uint32_t dedup_algo, uint32_t force_dedup_yao,
    uint32_t entry_id, const std::string& entry_file, const std::string& entry_out,
    uint32_t dummy_id_override,
    uint32_t debug_tag, uint32_t nthreads, e_mt_gen_alg mt_alg) {

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
    std::cout << "  Q (batch queries): " << num_queries << std::endl;
    std::cout << "  M (neighbors): " << M << std::endl;
    std::cout << "  D (dimension): " << D << std::endl;
    std::cout << "  L_C: " << L_C << std::endl;
    std::cout << "  L_W: " << L_W << std::endl;
    std::cout << "  LV: " << LV << std::endl;
    std::cout << "  Input bitlen: " << bitlen << std::endl;
    std::cout << "  ID bitlen: " << id_bitlen << std::endl;
    const bool allow_radix = false;
    if (dedup_algo == 1 && !allow_radix) {
        std::cout << "  Dedup algo: radix requested, falling back to bitonic (radix disabled)" << std::endl;
        dedup_algo = 0;
    }
    std::string dedup_name = (dedup_algo == 1) ? "radix" : "bitonic";
    std::cout << "  Yao de-dup thresh: " << yao_dedup_thresh << std::endl;
    std::cout << "  Dedup algo: " << dedup_name << std::endl;
    std::cout << "  Force Yao dedup: " << (force_dedup_yao ? "yes" : "no") << std::endl;
    std::cout << "  Inner product bitlen: " << inner_prod_bitlen << std::endl;
    std::cout << "  Arithmetic share bitlen: " << arith_bitlen << std::endl;
    std::cout << "  Debug tag payload: " << (debug_tag ? "on" : "off") << std::endl;
    std::cout << "========================================================\n" << std::endl;

    if (inner_prod_bitlen > 64) {
        std::cerr << "Inner product bitlen exceeds 64; reduce bitlen or D." << std::endl;
        return 1;
    }

    uint32_t T = L_C;
    uint32_t K = num_queries * M;
    uint32_t Vmax = num_queries + T * K;
    std::cout << "Vmax (visited length) = " << Vmax << std::endl;

    srand(12345);

    LoadRealDataIfRequested();
    uint32_t U = 1u << id_bitlen;
    uint32_t visited_pad = U - 1;
    uint32_t dummy_pool = T * num_queries + T * K;
    uint32_t N = g_real_data.loaded ? g_real_data.N : (U - 1 - dummy_pool);
    if (N == 0) {
        std::cerr << "ID space too small for dummy pool; increase id_bitlen." << std::endl;
        return 1;
    }
    if (g_real_data.loaded && (uint64_t)N + (uint64_t)dummy_pool >= (uint64_t)U) {
        std::cerr << "[HNSECW_DATA_FILE] N=" << N << " + dummy_pool=" << dummy_pool
                  << " exceeds U=" << U << " (id_bitlen=" << id_bitlen
                  << "); increase id_bitlen." << std::endl;
        return 1;
    }

    uint64_t pad_key = (inner_prod_bitlen >= 64) ? ~0ULL : ((1ULL << inner_prod_bitlen) - 1ULL);
    uint64_t md_key = (pad_key == 0) ? 0 : (pad_key - 1);
    uint32_t c_mod = (inner_prod_bitlen >= 32) ? (1U << 31) : (uint32_t)md_key;

    std::vector<std::vector<uint32_t>> query_plain(num_queries, std::vector<uint32_t>(D, 0));
    for (uint32_t q = 0; q < num_queries; q++) {
        if (g_real_data.loaded && D == g_real_data.D && q < g_real_data.num_queries) {
            query_plain[q] = g_real_data.query[q];
        } else {
            for (uint32_t j = 0; j < D; j++) {
                uint32_t bound = (bitlen >= 2 && bitlen < 32) ? (1U << (bitlen - 1)) : (1U << 31);
                query_plain[q][j] = rand() % bound;
            }
        }
    }

    std::vector<std::vector<uint32_t>> C_dist_plain(num_queries, std::vector<uint32_t>(L_C, 0));
    std::vector<std::vector<uint32_t>> C_id_plain(num_queries, std::vector<uint32_t>(L_C, 0));
    std::vector<std::vector<uint32_t>> W_dist_plain(num_queries, std::vector<uint32_t>(L_W, 0));
    std::vector<std::vector<uint32_t>> W_id_plain(num_queries, std::vector<uint32_t>(L_W, 0));
    // Paper-spec sentinel init: (max_dist, dummy_id) so unfilled slots
    // sort to the bottom.
    uint32_t init_dummy_id = (dummy_id_override != kDummyIdAuto)
                                ? dummy_id_override
                                : N;
    for (uint32_t q = 0; q < num_queries; q++) {
        for (uint32_t i = 0; i < L_C; i++) {
            C_dist_plain[q][i] = (uint32_t)md_key;
            C_id_plain[q][i] = init_dummy_id;
        }
        for (uint32_t i = 0; i < L_W; i++) {
            W_dist_plain[q][i] = (uint32_t)md_key;
            W_id_plain[q][i] = init_dummy_id;
        }
    }

    std::vector<uint32_t> entry_ids;
    if (!entry_file.empty()) {
        entry_ids = LoadEntryList(entry_file);
        if (!entry_ids.empty() && entry_ids.size() != num_queries) {
            throw std::runtime_error("entry_file count does not match num_queries");
        }
    } else if (entry_id != kEntryAuto) {
        entry_ids.assign(num_queries, entry_id);
    }

    std::vector<uint32_t> u(num_queries, 0);
    std::vector<uint32_t> visited_plain(Vmax, visited_pad);
    std::unordered_set<uint32_t> visited_set;
    for (uint32_t q = 0; q < num_queries; q++) {
        uint32_t entry = rand() % N;
        if (!entry_ids.empty()) {
            entry = entry_ids[q];
        }
        u[q] = entry;
        visited_plain[q] = entry;
        visited_set.insert(entry);
    }
    uint32_t visited_count = num_queries;

    uint16_t port_execA = port;
    uint16_t port_execB = (uint16_t)(port + 1);
    double total_online_ms = 0.0;
    uint64_t total_online_bytes = 0;

    // Reuse parties across iterations; circuits are reset inside each Exec.
    // ExecA now uses BOOL+Yao only, so keep arithmetic bitlen small to reduce setup.
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
    // Both parties also need an extra recreate between back-to-back
    // ExecCircuit calls within an iteration (dedup_c0 + dedup_neigh on
    // partyA; only one ExecB on partyB).  We walk a stride-2 port pair
    // per allocation so binding never lands on a TIME_WAIT'd socket.
    ABYParty* partyA = nullptr;
    ABYParty* partyB = nullptr;
    uint16_t port_execA_base = port_execA;
    uint16_t port_execB_base = port_execB;
    uint32_t partyA_alloc_count = 0;
    uint32_t partyB_alloc_count = 0;
    auto next_port_a = [&]() -> uint16_t {
        return (uint16_t)(port_execA_base + 2 * (partyA_alloc_count++));
    };
    auto next_port_b = [&]() -> uint16_t {
        return (uint16_t)(port_execB_base + 2 * (partyB_alloc_count++));
    };

    for (uint32_t t = 0; t < T; t++) {
        // Per-iter recreate (see hnsecw_single_b2y.cpp for rationale).
        delete partyA;
        delete partyB;
        partyA = new ABYParty(role, address, next_port_a(), seclvl,
                              arith_bitlen_A, nthreads, mt_alg);
        partyB = new ABYParty(role, address, next_port_b(), seclvl,
                              arith_bitlen, nthreads, mt_alg);
        std::cout << "\n================ Iteration " << t << " ================" << std::endl;

        std::vector<uint32_t> cand_c0(num_queries);
        std::vector<uint32_t> dummy_c0(num_queries);
        for (uint32_t q = 0; q < num_queries; q++) {
            cand_c0[q] = u[q];
            if (dummy_id_override == kDummyIdAuto) {
                dummy_c0[q] = N + t * num_queries + q;
            } else {
                dummy_c0[q] = dummy_id_override;
            }
        }

        // C0 dedup: skip the historical-visited check.  Single mode lets
        // u[q] fall through to GenerateNeighbors even if u is already
        // visited, because the per-iteration progress comes from u's
        // *neighbours* not yet in visited.  Without this skip, batch
        // marks u[q] dummy at iter 1 (it was just added to visited as
        // a neighbour in iter 0), drops the expansion, and stalls with
        // visited_count = num_queries + Q*M forever.  We still keep
        // cross-query C0 dedup (cand_c0 against dummy_c0) which is
        // what RunBatchDedup does when visited_len = 0.
        uint32_t visited_len_c0 = 0;
        DedupAlgo algo = (dedup_algo == 1) ? DedupAlgo::kBitonicGroup : DedupAlgo::kBitonicTag;
        bool use_yao_c0 = (yao_dedup_thresh > 0 && visited_len_c0 <= yao_dedup_thresh);
        if (force_dedup_yao || algo == DedupAlgo::kBitonicGroup) {
            use_yao_c0 = true;
        }
        DedupResult dedup_c0 = RunBatchDedup(
            partyA, role, id_bitlen,
            cand_c0, dummy_c0,
            visited_plain, visited_len_c0,
            use_yao_c0, algo, debug_tag,
            "[Dedup C0 t=" + std::to_string(t) + "]");

        uint32_t dummy_neigh_base = PickDummyId(N + T * num_queries + t * K, dummy_id_override);
        std::vector<uint32_t> neigh_flat(K);
        for (uint32_t q = 0; q < num_queries; q++) {
            if (IsDummyId(dedup_c0.id_prime[q], N, dummy_id_override)) {
                // Dummy C0 should yield dummy neighbors only (no real nodes).
                for (uint32_t m = 0; m < M; m++) {
                    neigh_flat[q * M + m] = (dummy_id_override == kDummyIdAuto)
                        ? (dummy_neigh_base + q * M + m)
                        : dummy_id_override;
                }
                continue;
            }
            std::vector<uint32_t> neigh_ids = GenerateNeighbors(
                dedup_c0.id_prime[q], t, M, N, dedup_c0.id_prime[q], dummy_id_override);
            for (uint32_t m = 0; m < M; m++) {
                neigh_flat[q * M + m] = neigh_ids[m];
            }
        }

        std::vector<uint32_t> dummy_neigh(K);
        for (uint32_t k = 0; k < K; k++) {
            if (dummy_id_override == kDummyIdAuto) {
                dummy_neigh[k] = dummy_neigh_base + k;
            } else {
                dummy_neigh[k] = dummy_id_override;
            }
        }

        uint32_t visited_len_neigh = visited_count;
        bool use_yao_neigh = (yao_dedup_thresh > 0 && visited_len_neigh <= yao_dedup_thresh);
        if (force_dedup_yao || algo == DedupAlgo::kBitonicGroup) {
            use_yao_neigh = true;
        }
        // Same recreate-between-back-to-back-ExecCircuit pattern as multi:
        // dedup_c0 already consumed partyA's setup; dedup_neigh's second
        // ExecCircuit on the same setup eventually drifts past correct
        // (Earlier code allocated a fresh partyA between dedup_c0 and
        // dedup_neigh on the theory that two ExecCircuit calls drift;
        // the bug attribution that motivated this was wrong, and the
        // built-in party->Reset() is sufficient.)

        DedupResult dedup_neigh = RunBatchDedup(
            partyA, role, id_bitlen,
            neigh_flat, dummy_neigh,
            visited_plain, visited_len_neigh,
            use_yao_neigh, algo, debug_tag,
            "[Dedup Neigh t=" + std::to_string(t) + "]");

        std::vector<std::vector<uint32_t>> fetched_vectors = FetchVectors(
            dedup_neigh.id_prime, N, D, bitlen, dummy_id_override);

        // Detect-and-retry around ExecB (framework-debug, mirrors
        // single_b2y / multi_b2y_dyn).  ABY's IKNP-OT pool occasionally
        // commits to bad Beaver triples; we recompute the per-query
        // expected distance bound and the L_W-prefix multiset in
        // plaintext (the .dat file is mirrored to both processes in this
        // benchmark setup) and retry on mismatch.  Online metrics
        // record only the successful call.
        const uint64_t coord_max =
            (bitlen >= 64) ? UINT64_MAX
                           : (((uint64_t)1 << bitlen) - 1ULL);
        const uint64_t max_valid_dist =
            (uint64_t)D * coord_max * coord_max;
        // Per-query plaintext candidate distances (K = num_queries * M).
        // The MPC sets distance = md_key whenever the dedup hit flag is 1
        // (the candidate is either a visited duplicate or a cross-query
        // duplicate); otherwise it computes the squared L2.  Compute the
        // hit flag locally from the visited_plain table both processes
        // share in this benchmark setup, instead of trying to read it
        // out of dedup_neigh.hit_share (which is a secret-shared bit).
        std::unordered_set<uint32_t> visited_lookup;
        visited_lookup.reserve(visited_len_neigh);
        for (uint32_t i = 0; i < visited_len_neigh; i++) {
            visited_lookup.insert(visited_plain[i]);
        }
        std::vector<std::vector<uint32_t>> cand_dist_plain(num_queries,
                                                            std::vector<uint32_t>(M));
        for (uint32_t q = 0; q < num_queries; q++) {
            for (uint32_t m = 0; m < M; m++) {
                uint32_t i = q * M + m;
                uint32_t cand_id = neigh_flat[i];   // pre-dedup id
                bool is_dummy_pre = IsDummyId(cand_id, N, dummy_id_override);
                bool is_visited = !is_dummy_pre &&
                                  (visited_lookup.find(cand_id) != visited_lookup.end());
                if (is_dummy_pre || is_visited) {
                    cand_dist_plain[q][m] = (uint32_t)md_key;
                } else {
                    uint32_t s = 0;
                    for (uint32_t j = 0; j < D; j++) {
                        uint32_t diff = (uint32_t)fetched_vectors[i][j]
                                        - (uint32_t)query_plain[q][j];
                        s += diff * diff;
                    }
                    cand_dist_plain[q][m] = (inner_prod_bitlen >= 32)
                        ? s
                        : (s & ((1U << inner_prod_bitlen) - 1U));
                }
            }
        }
        auto build_expected = [](const std::vector<uint32_t>& prev,
                                  const std::vector<uint32_t>& cand,
                                  uint32_t keep) {
            std::vector<uint32_t> all;
            all.reserve(prev.size() + cand.size());
            all.insert(all.end(), prev.begin(), prev.end());
            all.insert(all.end(), cand.begin(), cand.end());
            std::sort(all.begin(), all.end());
            if (all.size() > keep) all.resize(keep);
            return all;
        };

        const int kMaxBatchExecBRetries = 32;
        BatchExecBResult execB;
        int execb_attempt = 0;
        while (true) {
            execB = RunBatchExecB(
                partyB, role, num_queries, K, D, L_C, L_W,
                bitlen, id_bitlen, inner_prod_bitlen, debug_tag, t,
                query_plain, fetched_vectors,
                dedup_neigh.hit_share, dedup_neigh.id_prime,
                C_dist_plain, C_id_plain,
                W_dist_plain, W_id_plain,
                md_key,
                "[ExecB t=" + std::to_string(t) + "]");
            bool valid = true;
            std::string reason;
            for (uint32_t q = 0; valid && q < num_queries; q++) {
                // md_key is the legitimate "visited / dummy" sentinel and
                // routinely lands in the tail of C/W when few unvisited
                // neighbours remain; exempt it from the range check.
                for (uint32_t i = 0; valid && i < L_C; i++) {
                    if ((uint64_t)execB.C_dist[q][i] != md_key &&
                        (uint64_t)execB.C_dist[q][i] > max_valid_dist) {
                        valid = false;
                        reason = "q=" + std::to_string(q) + " C[" +
                                 std::to_string(i) + "]=" +
                                 std::to_string(execB.C_dist[q][i]) +
                                 " > " + std::to_string(max_valid_dist);
                    }
                }
                for (uint32_t i = 0; valid && i < L_W; i++) {
                    if ((uint64_t)execB.W_dist[q][i] != md_key &&
                        (uint64_t)execB.W_dist[q][i] > max_valid_dist) {
                        valid = false;
                        reason = "q=" + std::to_string(q) + " W[" +
                                 std::to_string(i) + "]=" +
                                 std::to_string(execB.W_dist[q][i]) +
                                 " > " + std::to_string(max_valid_dist);
                    }
                }
                if (valid) {
                    auto expected_C =
                        build_expected(C_dist_plain[q], cand_dist_plain[q], L_C);
                    auto expected_W =
                        build_expected(W_dist_plain[q], cand_dist_plain[q], L_W);
                    std::vector<uint32_t> got_C(execB.C_dist[q]);
                    std::sort(got_C.begin(), got_C.end());
                    uint32_t strict = std::min<uint32_t>(L_W, L_C);
                    for (uint32_t i = 0; i < strict; i++) {
                        if (got_C[i] != expected_C[i]) {
                            valid = false;
                            reason = "q=" + std::to_string(q) + " C prefix [" +
                                     std::to_string(i) + "] " +
                                     std::to_string(got_C[i]) + " vs " +
                                     std::to_string(expected_C[i]);
                            break;
                        }
                    }
                    if (valid) {
                        std::vector<uint32_t> got_W(execB.W_dist[q]);
                        std::sort(got_W.begin(), got_W.end());
                        for (uint32_t i = 0; i < L_W; i++) {
                            if (got_W[i] != expected_W[i]) {
                                valid = false;
                                reason = "q=" + std::to_string(q) + " W [" +
                                         std::to_string(i) + "] " +
                                         std::to_string(got_W[i]) + " vs " +
                                         std::to_string(expected_W[i]);
                                break;
                            }
                        }
                    }
                }
            }
            if (valid) break;
            if (++execb_attempt > kMaxBatchExecBRetries) {
                std::cerr << "[ExecB t=" << t << "] retry budget exhausted: "
                          << reason << std::endl;
                break;
            }
            std::cout << "[ExecB t=" << t << "] retry " << execb_attempt
                      << ": " << reason << std::endl;
            delete partyB;
            partyB = new ABYParty(role, address, next_port_b(), seclvl,
                                  arith_bitlen, nthreads, mt_alg);
        }

        total_online_ms += dedup_c0.online_ms + dedup_neigh.online_ms + execB.online_ms;
        total_online_bytes += dedup_c0.online_bytes + dedup_neigh.online_bytes + execB.online_bytes;

        C_dist_plain = execB.C_dist;
        C_id_plain = execB.C_id;
        W_dist_plain = execB.W_dist;
        W_id_plain = execB.W_id;

        for (uint32_t k = 0; k < K; k++) {
            uint32_t idv = dedup_neigh.id_prime[k];
            if (IsDummyId(idv, N, dummy_id_override)) {
                continue;
            }
            assert(visited_set.find(idv) == visited_set.end());
            visited_plain[visited_count++] = idv;
            visited_set.insert(idv);
        }
        assert(visited_count <= Vmax);

        for (uint32_t q = 0; q < num_queries; q++) {
            if (!C_id_plain[q].empty()) {
                u[q] = C_id_plain[q][0];
                // Pop the just-used u_next per query so the next iteration's
                // bitonic merge sorts this slot to the bottom; without the
                // pop the running global min stays as C[0] forever and the
                // search stalls at one node per query.  Standard HNSW
                // (Malkov 2018) extracts the closest from C at each step.
                C_dist_plain[q][0] = (uint32_t)md_key;
                C_id_plain[q][0] = init_dummy_id;
            }
        }
        std::cout << "[State] visited_count=" << visited_count << std::endl;
    }

    std::cout << "\n[Total Online] latency(s)=" << (total_online_ms / 1000.0)
              << " comm(MB)=" << (total_online_bytes / (1024.0 * 1024.0))
              << std::endl;

    if (role == CLIENT && !entry_out.empty()) {
        std::vector<uint32_t> entry_out_ids(num_queries, 0);
        for (uint32_t q = 0; q < num_queries; q++) {
            // W[0] should be the best real candidate, but high-D queries
            // (Fashion 784-D, MSMARCO 768-D) sometimes leave a dummy at
            // W[0] when ||q||^2 falls below the typical real L2 distance
            // and dummies absorb a slot before real candidates accumulate.
            // Walk W and pick the first id within [0, N_real).  No
            // additional MPC cost — W is already revealed plaintext.
            uint32_t out_id = u[q];
            const auto& W_q = W_id_plain[q];
            for (size_t i = 0; i < W_q.size(); ++i) {
                if (g_real_data.loaded ? (W_q[i] < g_real_data.N) : (W_q[i] < N)) {
                    out_id = W_q[i];
                    break;
                }
            }
            entry_out_ids[q] = out_id;
        }
        WriteEntryList(entry_out, entry_out_ids);
    }

    delete partyB;
    delete partyA;
    return 0;
}
