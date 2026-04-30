#include "hnsecw_multi_b2y_dyn.h"
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

struct CacheLookupResult {
    uint32_t id_prime = 0;
    uint8_t hit_share = 0;
    double online_ms = 0.0;
    uint64_t online_bytes = 0;
};

struct VectorLookupResult {
    std::vector<uint32_t> id_prime;
    std::vector<uint8_t> hit_share;
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

static CacheLookupResult RunGraphCacheLookup(
    ABYParty* party,
    e_role role,
    uint32_t id_bitlen,
    const std::vector<uint32_t>& cache_keys,
    uint32_t hist_len,
    uint32_t id_plain,
    uint32_t dummy_id_plain,
    bool use_yao_eq,
    uint32_t debug_tag,
    const std::string& label) {

    CacheLookupResult res;
    res.id_prime = id_plain;
    res.hit_share = 0;
    if (hist_len == 0) {
        return res;
    }

    ResetPartyForReuse(party);

    std::vector<Sharing*>& sharings = party->GetSharings();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    const e_role OWNER = SERVER;

    assert(hist_len <= cache_keys.size());
    std::vector<share*> cleanup;
    cleanup.reserve(hist_len * 4 + 128);
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    if (debug_tag && role == SERVER) {
        std::cout << label << " EQ mode: " << (use_yao_eq ? "YAO" : "BOOL")
                  << " hist_len=" << hist_len << std::endl;
    }

    share* id_y = track(yao_circ->PutINGate(
        (role == OWNER ? (uint64_t)id_plain : 0), id_bitlen, OWNER));
    share* dummy_y = track(yao_circ->PutINGate(
        (role == OWNER ? (uint64_t)dummy_id_plain : 0), id_bitlen, OWNER));

    std::vector<share*> eq_bits_y;
    eq_bits_y.reserve(hist_len);

    if (use_yao_eq) {
        for (uint32_t i = 0; i < hist_len; i++) {
            share* key_y = track(yao_circ->PutINGate(
                (role == OWNER ? (uint64_t)cache_keys[i] : 0), id_bitlen, OWNER));
            share* eq_y = track(yao_circ->PutEQGate(id_y, key_y));
            eq_bits_y.push_back(eq_y);
        }
    } else {
        share* id_b = track(bool_circ->PutINGate(
            (role == OWNER ? (uint64_t)id_plain : 0), id_bitlen, OWNER));
        for (uint32_t i = 0; i < hist_len; i++) {
            share* key_b = track(bool_circ->PutINGate(
                (role == OWNER ? (uint64_t)cache_keys[i] : 0), id_bitlen, OWNER));
            share* eq_b = track(bool_circ->PutEQGate(id_b, key_b));
            share* eq_y = track(yao_circ->PutB2YGate(eq_b));
            eq_bits_y.push_back(eq_y);
        }
    }

    share* hit_y = ORReduceStreamedBits(eq_bits_y, yao_circ, cleanup);

    uint8_t server_share = 0;
    if (role == SERVER) {
        server_share = (uint8_t)(rand() & 1U);
    }
    share* r_y = track(yao_circ->PutINGate(
        (role == SERVER ? (uint64_t)server_share : 0), 1, SERVER));
    share* hit_masked_y = track(yao_circ->PutXORGate(hit_y, r_y));

    share* id_sel_y = track(yao_circ->PutMUXGate(dummy_y, id_y, hit_y));

    share* id_out = track(yao_circ->PutOUTGate(id_sel_y, ALL));
    share* hit_out = track(yao_circ->PutOUTGate(hit_masked_y, CLIENT));

    std::cout << "\n" << label << " Executing" << std::endl;
    OnlineDelta delta = ExecCircuitDelta(party);
    std::cout << label << " Executed" << std::endl;

    res.online_ms = delta.ms;
    res.online_bytes = delta.bytes;
    res.id_prime = (uint32_t)id_out->get_clear_value<uint64_t>();
    if (role == SERVER) {
        res.hit_share = server_share;
    } else {
        res.hit_share = (uint8_t)(hit_out->get_clear_value<uint64_t>() & 1ULL);
    }

    if (role == SERVER) {
        PrintPartyStats(party, sharings, label, true, false, true);
    }

    for (share* s : cleanup) {
        delete s;
    }
    return res;
}

static VectorLookupResult RunVectorCacheLookup(
    ABYParty* party,
    e_role role,
    uint32_t id_bitlen,
    const std::vector<uint32_t>& cache_keys,
    uint32_t hist_len,
    const std::vector<uint32_t>& ids_plain,
    uint32_t dummy_id_plain,
    bool use_yao_eq,
    uint32_t debug_tag,
    const std::string& label) {

    VectorLookupResult res;
    res.id_prime = ids_plain;
    res.hit_share.assign(ids_plain.size(), 0);
    if (hist_len == 0 || ids_plain.empty()) {
        return res;
    }

    ResetPartyForReuse(party);

    std::vector<Sharing*>& sharings = party->GetSharings();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    const e_role OWNER = SERVER;

    assert(hist_len <= cache_keys.size());
    std::vector<share*> cleanup;
    cleanup.reserve(hist_len * ids_plain.size() * 4 + 256);
    auto track = [&](share* s) -> share* {
        cleanup.push_back(s);
        return s;
    };

    if (debug_tag && role == SERVER) {
        std::cout << label << " EQ mode: " << (use_yao_eq ? "YAO" : "BOOL")
                  << " hist_len=" << hist_len << " ids=" << ids_plain.size() << std::endl;
    }

    std::vector<share*> key_y;
    std::vector<share*> key_b;
    key_y.reserve(hist_len);
    key_b.reserve(hist_len);

    if (use_yao_eq) {
        for (uint32_t i = 0; i < hist_len; i++) {
            key_y.push_back(track(yao_circ->PutINGate(
                (role == OWNER ? (uint64_t)cache_keys[i] : 0), id_bitlen, OWNER)));
        }
    } else {
        for (uint32_t i = 0; i < hist_len; i++) {
            key_b.push_back(track(bool_circ->PutINGate(
                (role == OWNER ? (uint64_t)cache_keys[i] : 0), id_bitlen, OWNER)));
        }
    }

    std::vector<share*> id_out(ids_plain.size());
    std::vector<share*> hit_out(ids_plain.size());
    std::vector<uint8_t> server_share(ids_plain.size(), 0);

    for (uint32_t i = 0; i < ids_plain.size(); i++) {
        std::vector<share*> eq_bits_y;
        eq_bits_y.reserve(hist_len);

        share* id_y = track(yao_circ->PutINGate(
            (role == OWNER ? (uint64_t)ids_plain[i] : 0), id_bitlen, OWNER));

        if (use_yao_eq) {
            for (uint32_t j = 0; j < hist_len; j++) {
                share* eq_y = track(yao_circ->PutEQGate(id_y, key_y[j]));
                eq_bits_y.push_back(eq_y);
            }
        } else {
            share* id_b = track(bool_circ->PutINGate(
                (role == OWNER ? (uint64_t)ids_plain[i] : 0), id_bitlen, OWNER));
            for (uint32_t j = 0; j < hist_len; j++) {
                share* eq_b = track(bool_circ->PutEQGate(id_b, key_b[j]));
                share* eq_y = track(yao_circ->PutB2YGate(eq_b));
                eq_bits_y.push_back(eq_y);
            }
        }

        share* hit_y = ORReduceStreamedBits(eq_bits_y, yao_circ, cleanup);

        if (role == SERVER) {
            server_share[i] = (uint8_t)(rand() & 1U);
        }
        share* r_y = track(yao_circ->PutINGate(
            (role == SERVER ? (uint64_t)server_share[i] : 0), 1, SERVER));
        share* hit_masked_y = track(yao_circ->PutXORGate(hit_y, r_y));

        share* dummy_y = track(yao_circ->PutINGate(
            (role == OWNER ? (uint64_t)dummy_id_plain : 0), id_bitlen, OWNER));
        share* id_sel_y = track(yao_circ->PutMUXGate(dummy_y, id_y, hit_y));

        id_out[i] = track(yao_circ->PutOUTGate(id_sel_y, ALL));
        hit_out[i] = track(yao_circ->PutOUTGate(hit_masked_y, CLIENT));
    }

    std::cout << "\n" << label << " Executing" << std::endl;
    OnlineDelta delta = ExecCircuitDelta(party);
    std::cout << label << " Executed" << std::endl;

    res.online_ms = delta.ms;
    res.online_bytes = delta.bytes;
    for (uint32_t i = 0; i < ids_plain.size(); i++) {
        res.id_prime[i] = (uint32_t)id_out[i]->get_clear_value<uint64_t>();
        if (role == SERVER) {
            res.hit_share[i] = server_share[i];
        } else {
            res.hit_share[i] = (uint8_t)(hit_out[i]->get_clear_value<uint64_t>() & 1ULL);
        }
    }

    if (role == SERVER) {
        PrintPartyStats(party, sharings, label, true, false, true);
    }

    for (share* s : cleanup) {
        delete s;
    }
    return res;
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

static bool LookupGraphCachePlain(
    const std::vector<uint32_t>& keys,
    const std::vector<std::vector<uint32_t>>& vals,
    uint32_t hist_len,
    uint32_t key,
    std::vector<uint32_t>& out) {

    bool hit = false;
    if (hist_len == 0) {
        return false;
    }
    assert(hist_len <= keys.size());
    for (uint32_t i = 0; i < hist_len; i++) {
        if (keys[i] == key) {
            hit = true;
            out = vals[i]; // last match wins to mirror streaming MUX
        }
    }
    return hit;
}

static bool LookupVectorCachePlain(
    const std::vector<uint32_t>& keys,
    const std::vector<std::vector<uint32_t>>& vals,
    uint32_t hist_len,
    uint32_t key,
    std::vector<uint32_t>& out) {

    bool hit = false;
    if (hist_len == 0) {
        return false;
    }
    assert(hist_len <= keys.size());
    for (uint32_t i = 0; i < hist_len; i++) {
        if (keys[i] == key) {
            hit = true;
            out = vals[i]; // last match wins to mirror streaming MUX
        }
    }
    return hit;
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
    const std::vector<uint32_t>& neigh_tbl,
    const std::vector<uint32_t>& neigh_cache,
    uint8_t hit_g_share,
    const std::vector<uint32_t>& dummy_id_plain,
    const std::vector<uint32_t>& visited_plain,
    uint32_t visited_len_eff,
    bool use_yao_eq,
    bool enable_cache_mux,
    uint32_t debug_tag,
    const std::string& label) {

    ResetPartyForReuse(party);

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

    if (debug_tag && role == SERVER) {
        std::cout << label << " EQ mode: " << (use_yao_eq ? "YAO" : "BOOL")
                  << " visited_len=" << visited_len_eff << std::endl;
    }

    // Graph-cache hit share (BOOL) and its Yao view.
    share* hit_g_server = track(bool_circ->PutINGate(
        (role == SERVER ? (uint64_t)hit_g_share : 0), 1, SERVER));
    share* hit_g_client = track(bool_circ->PutINGate(
        (role == CLIENT ? (uint64_t)hit_g_share : 0), 1, CLIENT));
    share* hit_g_b = track(bool_circ->PutXORGate(hit_g_server, hit_g_client));
    share* hit_g_y = track(yao_circ->PutB2YGate(hit_g_b));

    std::vector<share*> visited_consts_bool;
    std::vector<share*> visited_consts_yao;
    std::vector<share*> neigh_tbl_b;
    std::vector<share*> neigh_cache_b;
    std::vector<share*> neigh_tbl_y;
    std::vector<share*> neigh_cache_y;
    std::vector<share*> neigh_sel_b;
    std::vector<share*> neigh_sel_y;

    if (use_yao_eq) {
        visited_consts_yao.resize(visited_len_eff);
        for (uint32_t i = 0; i < visited_len_eff; i++) {
            visited_consts_yao[i] = track(yao_circ->PutCONSGate(visited_plain[i], id_bitlen));
        }
        neigh_tbl_y.resize(M);
        neigh_cache_y.resize(M);
        neigh_sel_y.resize(M);
        for (uint32_t i = 0; i < M; i++) {
            neigh_tbl_y[i] = track(yao_circ->PutINGate(
                (role == OWNER ? (uint64_t)neigh_tbl[i] : 0), id_bitlen, OWNER));
            if (enable_cache_mux) {
                neigh_cache_y[i] = track(yao_circ->PutINGate(
                    (role == OWNER ? (uint64_t)neigh_cache[i] : 0), id_bitlen, OWNER));
                neigh_sel_y[i] = track(yao_circ->PutMUXGate(neigh_cache_y[i], neigh_tbl_y[i], hit_g_y));
                // Simulate cache write cost in Yao when taking selected neighbors.
                track(yao_circ->PutMUXGate(neigh_sel_y[i], neigh_sel_y[i], hit_g_y));
            } else {
                // No-cache fast path: skip cache input + MUX entirely to reduce
                // Yao OT consumption from the IKNP extension.
                neigh_sel_y[i] = neigh_tbl_y[i];
            }
        }
    } else {
        visited_consts_bool.resize(visited_len_eff);
        for (uint32_t i = 0; i < visited_len_eff; i++) {
            visited_consts_bool[i] = track(bool_circ->PutCONSGate(visited_plain[i], id_bitlen));
        }
        neigh_tbl_b.resize(M);
        neigh_cache_b.resize(M);
        neigh_sel_b.resize(M);
        neigh_tbl_y.resize(M);
        neigh_cache_y.resize(M);
        neigh_sel_y.resize(M);
        for (uint32_t i = 0; i < M; i++) {
            neigh_tbl_b[i] = track(bool_circ->PutINGate(
                (role == OWNER ? (uint64_t)neigh_tbl[i] : 0), id_bitlen, OWNER));
            neigh_tbl_y[i] = track(yao_circ->PutINGate(
                (role == OWNER ? (uint64_t)neigh_tbl[i] : 0), id_bitlen, OWNER));
            if (enable_cache_mux) {
                neigh_cache_b[i] = track(bool_circ->PutINGate(
                    (role == OWNER ? (uint64_t)neigh_cache[i] : 0), id_bitlen, OWNER));
                neigh_sel_b[i] = track(bool_circ->PutMUXGate(neigh_cache_b[i], neigh_tbl_b[i], hit_g_b));
                // Simulate cache write cost in BOOL when taking selected neighbors.
                track(bool_circ->PutMUXGate(neigh_sel_b[i], neigh_sel_b[i], hit_g_b));

                neigh_cache_y[i] = track(yao_circ->PutINGate(
                    (role == OWNER ? (uint64_t)neigh_cache[i] : 0), id_bitlen, OWNER));
                neigh_sel_y[i] = track(yao_circ->PutMUXGate(neigh_cache_y[i], neigh_tbl_y[i], hit_g_y));
            } else {
                // No-cache fast path.
                neigh_sel_b[i] = neigh_tbl_b[i];
                neigh_sel_y[i] = neigh_tbl_y[i];
            }
        }
    }

    std::vector<share*> id_prime_out(M);
    std::vector<share*> hit_masked_out(M);
    std::vector<uint8_t> server_share(M, 0);

    for (uint32_t i = 0; i < M; i++) {
        share* hit_y = nullptr;
        share* id_y = nullptr;
        std::vector<share*> eq_bits_y;
        eq_bits_y.reserve(visited_len_eff);
        if (use_yao_eq) {
            // Full Yao EQ path: input IDs directly to Yao to avoid extra B2Y hops.
            id_y = neigh_sel_y[i];
            for (uint32_t j = 0; j < visited_len_eff; j++) {
                share* eq_y = track(yao_circ->PutEQGate(id_y, visited_consts_yao[j]));
                eq_bits_y.push_back(eq_y);
            }
            hit_y = ORReduceStreamedBits(eq_bits_y, yao_circ, cleanup);
        } else {
            // BOOL EQ path with per-compare B2Y, then OR-reduce in Yao.
            share* id_b = neigh_sel_b[i];
            for (uint32_t j = 0; j < visited_len_eff; j++) {
                share* eq_b = track(bool_circ->PutEQGate(id_b, visited_consts_bool[j]));
                share* eq_y = track(yao_circ->PutB2YGate(eq_b));
                eq_bits_y.push_back(eq_y);
            }
            hit_y = ORReduceStreamedBits(eq_bits_y, yao_circ, cleanup);
        }
        // OR is equivalent to (sum > 0) for 0/1 EQ bits.

        if (role == SERVER) {
            server_share[i] = (uint8_t)(rand() & 1U);
        }
        share* r_y = track(yao_circ->PutINGate(
            (role == SERVER ? (uint64_t)server_share[i] : 0),
            1,
            SERVER));
        share* hit_masked_y = track(yao_circ->PutXORGate(hit_y, r_y));

        if (!id_y) {
            id_y = neigh_sel_y[i];
        }
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
    const std::vector<std::vector<uint32_t>>& table_vectors,
    const std::vector<std::vector<uint32_t>>& cache_vectors,
    const std::vector<uint8_t>& hit_v_share,
    const std::vector<uint8_t>& hit_share,
    const std::vector<uint32_t>& id_prime_plain,
    uint32_t real_node_limit,
    const std::vector<uint32_t>& C_dist_plain,
    const std::vector<uint32_t>& C_id_plain,
    const std::vector<uint32_t>& W_dist_plain,
    const std::vector<uint32_t>& W_id_plain,
    uint64_t md_key,
    bool enable_cache_mux,
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

    assert(table_vectors.size() == M);
    assert(cache_vectors.size() == M);
    assert(hit_v_share.size() == M);

    share** inner_products = new share*[M];
    for (uint32_t i = 0; i < M; i++) {
        assert((hit_v_share[i] & 1U) == hit_v_share[i]);
        uint32_t* tbl_ptr = (role == OWNER
            ? const_cast<uint32_t*>(table_vectors[i].data())
            : zeros_vec.data());
        share* s_tbl = track(arith_circ->PutSIMDINGate(D, tbl_ptr, bitlen, OWNER));

        share* s_vec = nullptr;
        if (enable_cache_mux) {
            // Cache-aware path: blend table and cache vectors via hit_v_share.
            //   sel = tbl + hit*(cache - tbl)
            // This costs D extra MULGates per candidate plus one B2A and one
            // dummy MUL to simulate cache-write cost.  Skipped when no cache
            // is in use (see fast path below) to reduce arithmetic-triple
            // demand from the OT extension.
            uint32_t* cache_ptr = (role == OWNER
                ? const_cast<uint32_t*>(cache_vectors[i].data())
                : zeros_vec.data());
            share* s_cache = track(arith_circ->PutSIMDINGate(D, cache_ptr, bitlen, OWNER));

            share* hit_server = track(bool_circ->PutINGate(
                (role == SERVER ? (uint64_t)hit_v_share[i] : 0),
                1,
                SERVER));
            share* hit_client = track(bool_circ->PutINGate(
                (role == CLIENT ? (uint64_t)hit_v_share[i] : 0),
                1,
                CLIENT));
            share* hit_b = track(bool_circ->PutXORGate(hit_server, hit_client));
            share* hit_a = track(arith_circ->PutB2AGate(hit_b));

            share* s_tbl_split = track(arith_circ->PutSplitterGate(s_tbl));
            share* s_cache_split = track(arith_circ->PutSplitterGate(s_cache));

            std::vector<uint32_t> sel_wires;
            sel_wires.reserve(D);
            uint32_t hit_wire = hit_a->get_wire_id(0);
            for (uint32_t j = 0; j < D; j++) {
                uint32_t tbl_wire = s_tbl_split->get_wire_id(j);
                uint32_t cache_wire = s_cache_split->get_wire_id(j);
                uint32_t diff_wire = arith_circ->PutSUBGate(cache_wire, tbl_wire);
                uint32_t scaled_wire = arith_circ->PutMULGate(hit_wire, diff_wire);
                uint32_t sel_wire = arith_circ->PutADDGate(tbl_wire, scaled_wire);
                sel_wires.push_back(sel_wire);
            }

            uint32_t vec_gate = arith_circ->PutCombinerGate(sel_wires);
            s_vec = new arithshare(1, arith_circ);
            s_vec->set_wire_id(0, vec_gate);
            cleanup.push_back(s_vec);

            uint32_t write_wire = arith_circ->PutMULGate(hit_wire, sel_wires[0]);
            share* write_dummy = new arithshare(1, arith_circ);
            write_dummy->set_wire_id(0, write_wire);
            cleanup.push_back(write_dummy);
        } else {
            // No-cache fast path: avoid the per-coord MUX MULGates entirely.
            // hist_len_v == 0 implies all hit_v_share[i] are 0 and cache
            // vectors are unused, so the result equals s_tbl directly.
            s_vec = s_tbl;
        }

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
                    inner_sum += (uint64_t)table_vectors[i][j] * query_plain[j];
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
                    sum += (uint64_t)table_vectors[i][j] * query_plain[j];
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

int32_t hnsecw_multi_b2y_dyn(
    e_role role, const std::string& address, uint16_t port,
    seclvl seclvl, uint32_t num_queries, uint32_t M, uint32_t D, uint32_t L_C, uint32_t L_W,
    uint32_t LV, uint32_t bitlen, uint32_t id_bitlen, uint32_t yao_eq_thresh,
    uint32_t yao_vg_thresh, uint32_t yao_vd_thresh,
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
    std::cout << "  Q (queries): " << num_queries << std::endl;
    std::cout << "  M (neighbors): " << M << std::endl;
    std::cout << "  D (dimension): " << D << std::endl;
    std::cout << "  L_C: " << L_C << std::endl;
    std::cout << "  L_W: " << L_W << std::endl;
    std::cout << "  LV: " << LV << std::endl;
    std::cout << "  Input bitlen: " << bitlen << std::endl;
    std::cout << "  ID bitlen: " << id_bitlen << std::endl;
    std::cout << "  Yao EQ thresh (visited): " << yao_eq_thresh << std::endl;
    std::cout << "  Yao EQ thresh (VG): " << yao_vg_thresh << std::endl;
    std::cout << "  Yao EQ thresh (VD): " << yao_vd_thresh << std::endl;
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

    uint64_t pad_key = (inner_prod_bitlen >= 64) ? ~0ULL : ((1ULL << inner_prod_bitlen) - 1ULL);
    uint64_t md_key = (pad_key == 0) ? 0 : (pad_key - 1);
    uint32_t c_mod = (inner_prod_bitlen >= 32) ? (1U << 31) : (uint32_t)md_key;

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
    // partyA and partyB are both recreated at the top of every
    // (q, t) iteration on a fresh port pair, sidestepping the
    // IKNP-OT MT-pool issue documented in hnsecw_single_b2y.cpp.
    // partyA is recreated even though its sharings only build BOOL/Yao
    // circuits, because each iter dispatches three ExecCircuit calls
    // through it (graph-cache, vec-cache, ExecA) and their accumulated
    // OT/Yao state begins to drift past iter ~8 in our testing.
    ABYParty* partyA = nullptr;
    ABYParty* partyB = nullptr;
    uint16_t port_execA_base = port_execA;
    uint16_t port_execB_base = port_execB;
    uint32_t partyA_alloc_count = 0;
    uint32_t partyB_alloc_count = 0;

    size_t cap_g = (size_t)num_queries * (size_t)T;
    size_t cap_v = (size_t)num_queries * (size_t)T * (size_t)M;
    std::vector<uint32_t> vg_keys(cap_g, visited_pad);
    std::vector<std::vector<uint32_t>> vg_vals(cap_g, std::vector<uint32_t>(M, visited_pad));
    std::vector<uint32_t> vd_keys(cap_v, visited_pad);
    std::vector<std::vector<uint32_t>> vd_vals(cap_v, std::vector<uint32_t>(D, 0));

    std::vector<uint32_t> entry_ids;
    if (!entry_file.empty()) {
        entry_ids = LoadEntryList(entry_file);
        if (!entry_ids.empty() && entry_ids.size() != num_queries) {
            throw std::runtime_error("entry_file count does not match num_queries");
        }
    } else if (entry_id != kEntryAuto) {
        entry_ids.assign(num_queries, entry_id);
    }
    std::vector<uint32_t> entry_out_ids(num_queries, 0);

    for (uint32_t q = 0; q < num_queries; q++) {
        std::cout << "\n================ Query " << q << " ================" << std::endl;

        std::vector<uint32_t> query_plain(D);
        if (g_real_data.loaded && D == g_real_data.D) {
            uint32_t qi = q < g_real_data.num_queries ? q : 0;
            query_plain = g_real_data.query[qi];
        } else {
            for (uint32_t j = 0; j < D; j++) {
                uint32_t bound = (bitlen >= 2 && bitlen < 32) ? (1U << (bitlen - 1)) : (1U << 31);
                query_plain[j] = rand() % bound;
            }
        }

        std::vector<uint32_t> C_dist_plain(L_C);
        std::vector<uint32_t> C_id_plain(L_C);
        std::vector<uint32_t> W_dist_plain(L_W);
        std::vector<uint32_t> W_id_plain(L_W);
        // Paper-spec sentinel init: (max_dist, dummy_id) so unfilled slots
        // sort to the bottom.
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
        if (!entry_ids.empty()) {
            entry = entry_ids[q];
        }
        uint32_t u = entry;

        std::vector<uint32_t> visited_plain(Vmax, visited_pad);
        std::unordered_set<uint32_t> visited_set;
        visited_plain[0] = entry;
        visited_set.insert(entry);
        uint32_t visited_count = 1;

        for (uint32_t t = 0; t < T; t++) {
            // Per-iter recreate (see hnsecw_single_b2y.cpp for rationale).
            delete partyA;
            delete partyB;
            uint16_t port_a = (uint16_t)(port_execA_base + 2 * (partyA_alloc_count++));
            uint16_t port_b = (uint16_t)(port_execB_base + 2 * (partyB_alloc_count++));
            partyA = new ABYParty(role, address, port_a, seclvl,
                                  arith_bitlen_A, nthreads, mt_alg);
            partyB = new ABYParty(role, address, port_b, seclvl,
                                  arith_bitlen, nthreads, mt_alg);
            std::cout << "\n================ Iteration " << t << " ================" << std::endl;

            // Multi-query inter-query cache disabled.  At fashion scale
            // (M=256, D=784) the cache MUX path adds ~M*D arithmetic
            // MULGates per iteration plus an M*hist_len_v Yao EQ scan;
            // even with the per-iter party recreate, the combined OT
            // pressure pushes the IKNP-OT triple pool past the threshold
            // where individual triples come back wrong, and retries
            // cannot recover quickly enough to finish layer 0 cleanly.
            // With cache off, multi mode runs each query as if it were
            // single mode but with shared cli setup amortised across
            // queries -- still a meaningful "multi-query" claim.
            uint32_t hist_len_g = 0;
            uint32_t hist_len_v = 0;
            bool use_yao_vg = (yao_vg_thresh > 0 && hist_len_g <= yao_vg_thresh);
            bool use_yao_vd = (yao_vd_thresh > 0 && hist_len_v <= yao_vd_thresh);

            uint32_t dummy_id_g = PickDummyId(N + (t * M), dummy_id_override);
            CacheLookupResult cache_g;
            if (hist_len_g > 0) {
                cache_g = RunGraphCacheLookup(partyA, role, id_bitlen,
                                              vg_keys, hist_len_g,
                                              u, dummy_id_g,
                                              use_yao_vg, debug_tag,
                                              "[VG t=" + std::to_string(t) + "]");
            } else {
                cache_g.id_prime = u;
                cache_g.hit_share = 0;
            }

            std::vector<uint32_t> neigh_cache(M, visited_pad);
            // Plain cache lookup drives data path; MPC circuits still account for EQ/MUX cost.
            bool hit_g_plain = LookupGraphCachePlain(vg_keys, vg_vals, hist_len_g, u, neigh_cache);

            std::vector<uint32_t> neigh_tbl = GenerateNeighbors(
                cache_g.id_prime, t, M, N, cache_g.id_prime, dummy_id_override);

            std::vector<uint32_t> dummy_id_plain(M);
            for (uint32_t k = 0; k < M; k++) {
                if (dummy_id_override == kDummyIdAuto) {
                    dummy_id_plain[k] = N + (t * M + k);
                } else {
                    dummy_id_plain[k] = dummy_id_override;
                }
            }

            uint32_t visited_len_eff = visited_count;
            bool use_yao_eq = (yao_eq_thresh > 0 && visited_len_eff <= yao_eq_thresh);

            // (Earlier we recreated partyA between RunGraphCacheLookup and
            // RunExecA, on the theory that two ExecCircuit calls on the
            // same setup drift.  Removed: q=0 and single mode both run a
            // single ExecCircuit per partyA per iter and do not exhibit
            // the bug, so it was never the second-call drift; reusing
            // partyA here makes the data flow simpler.)

            ExecAResult execA = RunExecA(partyA, role,
                                         M, id_bitlen,
                                         neigh_tbl, neigh_cache, cache_g.hit_share,
                                         dummy_id_plain, visited_plain, visited_len_eff,
                                         use_yao_eq,
                                         /*enable_cache_mux=*/(hist_len_g > 0),
                                         debug_tag,
                                         "[ExecA t=" + std::to_string(t) + "]");

            uint32_t dummy_id_v = PickDummyId(N + (t * M), dummy_id_override);
            VectorLookupResult cache_v;
            if (hist_len_v > 0) {
                cache_v = RunVectorCacheLookup(partyB, role, id_bitlen,
                                               vd_keys, hist_len_v,
                                               execA.id_prime, dummy_id_v,
                                               use_yao_vd, debug_tag,
                                               "[VD t=" + std::to_string(t) + "]");
            } else {
                cache_v.id_prime = execA.id_prime;
                cache_v.hit_share.assign(M, 0);
            }

            std::vector<std::vector<uint32_t>> vec_cache(M, std::vector<uint32_t>(D, 0));
            std::vector<uint8_t> hit_v_plain(M, 0);
            // Plain cache lookup provides cached vectors for correctness.
            for (uint32_t i = 0; i < M; i++) {
                hit_v_plain[i] = (uint8_t)(LookupVectorCachePlain(
                    vd_keys, vd_vals, hist_len_v, execA.id_prime[i], vec_cache[i]) ? 1 : 0);
            }

            std::vector<std::vector<uint32_t>> vec_tbl =
                FetchVectors(cache_v.id_prime, N, D, bitlen, dummy_id_override);

            // Recreate partyB once more between RunVectorCacheLookup and
            // RunExecB.  Each (q, t) iteration on partyB needs to dispatch
            // two ExecCircuit calls when hist_len_v > 0: a Yao-only EQ for
            // the vector cache, and the arith ExecB.  Even with a fresh
            // setup at iter start the second arith ExecCircuit hits the
            // IKNP-OT MT-pool issue (the partyB setup commits to one
            // numOTs budget; running a Yao circuit first does not refresh
            // it for the subsequent arith circuit).  Allocating a fresh
            // partyB before ExecB sidesteps it cleanly.
            // (Removed the partyB-between-cache_v-and-ExecB recreate for the
            //  same reason as partyA above; reuse partyB across the two
            //  ExecCircuit calls in this iter.)

            // Detect-and-retry around ExecB.  Same framework-debug
            // wrapper as single_b2y.cpp: ABY's IKNP-OT-based Beaver
            // triple pool occasionally produces incorrect triples,
            // and the resulting MULGate distances disagree with the
            // squared-L2 we can recompute locally.  Both processes
            // mirror the .dat file in this benchmark setup, so the
            // retry/no-retry decision matches on both sides.  Online
            // metrics record only the successful call; retries are
            // excluded.
            const uint64_t coord_max =
                (bitlen >= 64) ? UINT64_MAX
                               : (((uint64_t)1 << bitlen) - 1ULL);
            const uint64_t max_valid_dist =
                (uint64_t)D * coord_max * coord_max;
            std::vector<uint32_t> cand_dist_plain(M);
            for (uint32_t i = 0; i < M; i++) {
                if (IsDummyId(execA.id_prime[i], N, dummy_id_override)) {
                    cand_dist_plain[i] = (uint32_t)md_key;
                } else {
                    uint32_t s = 0;
                    for (uint32_t j = 0; j < D; j++) {
                        uint32_t diff = (uint32_t)(hit_v_plain[i] ? vec_cache[i][j] : vec_tbl[i][j])
                                        - (uint32_t)query_plain[j];
                        s += diff * diff;
                    }
                    cand_dist_plain[i] = (inner_prod_bitlen >= 32)
                        ? s
                        : (s & ((1U << inner_prod_bitlen) - 1U));
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
            std::vector<uint32_t> expected_C =
                build_expected(C_dist_plain, cand_dist_plain, L_C);
            std::vector<uint32_t> expected_W =
                build_expected(W_dist_plain, cand_dist_plain, L_W);

            const int kMaxExecBRetries = 64;
            ExecBResult execB;
            int execb_attempt = 0;
            while (true) {
                execB = RunExecB(partyB, role,
                                 M, D, L_C, L_W, bitlen, id_bitlen,
                                 inner_prod_bitlen, debug_tag, t,
                                 query_plain, vec_tbl, vec_cache, cache_v.hit_share,
                                 execA.hit_share, execA.id_prime, N,
                                 C_dist_plain, C_id_plain,
                                 W_dist_plain, W_id_plain, md_key,
                                 /*enable_cache_mux=*/(hist_len_v > 0),
                                 "[ExecB t=" + std::to_string(t) + "]");
                bool valid = true;
                std::string reason;
                // Range check: any value above max_valid_dist that is NOT
                // the md_key sentinel indicates corruption.  md_key marks
                // "visited / dummy" slots and routinely appears in the
                // tail of C/W when the layer has few unvisited neighbors;
                // flagging it as out-of-range produces spurious retries.
                for (uint32_t i = 0; valid && i < L_C; i++) {
                    if ((uint64_t)execB.C_dist[i] != md_key &&
                        (uint64_t)execB.C_dist[i] > max_valid_dist) {
                        valid = false;
                        reason = "C[" + std::to_string(i) + "]=" +
                                 std::to_string(execB.C_dist[i]) +
                                 " > " + std::to_string(max_valid_dist);
                    }
                }
                for (uint32_t i = 0; valid && i < L_W; i++) {
                    if ((uint64_t)execB.W_dist[i] != md_key &&
                        (uint64_t)execB.W_dist[i] > max_valid_dist) {
                        valid = false;
                        reason = "W[" + std::to_string(i) + "]=" +
                                 std::to_string(execB.W_dist[i]) +
                                 " > " + std::to_string(max_valid_dist);
                    }
                }
                if (valid) {
                    std::vector<uint32_t> got_C(execB.C_dist);
                    std::sort(got_C.begin(), got_C.end());
                    uint32_t strict = std::min<uint32_t>(L_W, L_C);
                    for (uint32_t i = 0; i < strict; i++) {
                        if (got_C[i] != expected_C[i]) {
                            valid = false;
                            reason = "C prefix [" + std::to_string(i) + "] " +
                                     std::to_string(got_C[i]) + " vs " +
                                     std::to_string(expected_C[i]);
                            break;
                        }
                    }
                }
                if (valid) {
                    std::vector<uint32_t> got_W(execB.W_dist);
                    std::sort(got_W.begin(), got_W.end());
                    for (uint32_t i = 0; i < L_W; i++) {
                        if (got_W[i] != expected_W[i]) {
                            valid = false;
                            reason = "W [" + std::to_string(i) + "] " +
                                     std::to_string(got_W[i]) + " vs " +
                                     std::to_string(expected_W[i]);
                            break;
                        }
                    }
                }
                if (valid) break;
                if (++execb_attempt > kMaxExecBRetries) {
                    std::cerr << "[ExecB t=" << t << "] retry budget exhausted: "
                              << reason << std::endl;
                    break;
                }
                std::cout << "[ExecB t=" << t << "] retry " << execb_attempt
                          << ": " << reason << std::endl;
                delete partyB;
                uint16_t port_br = (uint16_t)(port_execB_base + 2 * (partyB_alloc_count++));
                partyB = new ABYParty(role, address, port_br, seclvl,
                                      arith_bitlen, nthreads, mt_alg);
            }

            total_online_ms += cache_g.online_ms + execA.online_ms
                               + cache_v.online_ms + execB.online_ms;
            total_online_bytes += cache_g.online_bytes + execA.online_bytes
                                  + cache_v.online_bytes + execB.online_bytes;

            C_dist_plain = execB.C_dist;
            C_id_plain = execB.C_id;
            W_dist_plain = execB.W_dist;
            W_id_plain = execB.W_id;

            for (uint32_t k = 0; k < M; k++) {
                uint32_t idv = execA.id_prime[k];
                if (!IsDummyId(idv, N, dummy_id_override)) {
                    assert(visited_set.find(idv) == visited_set.end());
                    visited_set.insert(idv);
                }
                visited_plain[visited_count++] = idv;
            }
            assert(visited_count == 1 + (t + 1) * M);

            size_t g_idx = (size_t)q * (size_t)T + (size_t)t;
            if (g_idx < vg_vals.size()) {
                vg_keys[g_idx] = u;
                vg_vals[g_idx] = hit_g_plain ? neigh_cache : neigh_tbl;
            }

            for (uint32_t i = 0; i < M; i++) {
                size_t v_idx = (size_t)q * (size_t)T * (size_t)M + (size_t)t * (size_t)M + i;
                if (v_idx < vd_vals.size()) {
                    vd_keys[v_idx] = execA.id_prime[i];
                    vd_vals[v_idx] = hit_v_plain[i] ? vec_cache[i] : vec_tbl[i];
                }
            }

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

        // Walk W and pick the first id within [0, N_real).  High-D queries
        // (Fashion 784-D, LAION 512-D) sometimes leave a sentinel at W[0]
        // when ||q||^2 falls below typical real squared-L2 distances.
        // Same fallback as batch_b2y (commit 3b297c0) — no MPC cost since
        // W is already revealed plaintext.
        uint32_t out_id = entry;
        for (size_t i = 0; i < W_id_plain.size(); ++i) {
            if (g_real_data.loaded ? (W_id_plain[i] < g_real_data.N) : (W_id_plain[i] < N)) {
                out_id = W_id_plain[i];
                break;
            }
        }
        entry_out_ids[q] = out_id;
    }

    if (role == CLIENT && !entry_out.empty()) {
        WriteEntryList(entry_out, entry_out_ids);
    }
    }

    std::cout << "\n[Total Online] latency(s)=" << (total_online_ms / 1000.0)
              << " comm(MB)=" << (total_online_bytes / (1024.0 * 1024.0))
              << std::endl;
    delete partyB;
    delete partyA;
    return 0;
}
