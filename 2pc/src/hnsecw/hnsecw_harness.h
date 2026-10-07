// Benchmark harness shared by the single, multi and batch programs:
// layer tables, cost accounting, and small circuit helpers.
#ifndef HNSECW_HARNESS_H
#define HNSECW_HARNESS_H

#include "hnsecw_core.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace hnsecw {

constexpr uint32_t kEntryAuto = std::numeric_limits<uint32_t>::max();

inline uint32_t CeilLog2(uint32_t n) {
    uint32_t k = 0;
    uint32_t p = 1;
    while (p < n) {
        p <<= 1;
        k++;
    }
    return k;
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

inline std::vector<uint32_t> ReadIdsFromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("failed to open " + path);
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint32_t> ids((size_t)size / sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(uint32_t));
    return ids;
}

inline void WriteIdsToFile(const std::string& path, const std::vector<uint32_t>& ids) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("failed to open " + path);
    out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// Layer tables
// ---------------------------------------------------------------------------
//
// With HNSECW_DATA_FILE set, the graph, the vectors and the queries of the
// layer come from that file (bench/run_search.py writes it):
//
//     u32 N, u32 D, u32 M, u32 Q
//     N*M u32  neighbour ids, unused slots hold a value >= N
//     N*D u32  coordinates
//     Q*D u32  query coordinates
//
// Without it the layer is a synthetic one of the requested shape.  The cost of
// the protocol depends only on the public parameters, so a synthetic layer is
// enough to measure it.

struct LayerData {
    bool loaded = false;
    uint32_t N = 0;
    uint32_t D = 0;
    uint32_t M = 0;
    uint32_t num_queries = 0;
    std::vector<uint32_t> graph;  // N * M
    std::vector<uint32_t> vec;    // N * D
    std::vector<uint32_t> query;  // num_queries * D
};

inline LayerData& Data() {
    static LayerData data;
    return data;
}

inline void LoadLayerData() {
    LayerData& g = Data();
    if (g.loaded) return;
    const char* env = std::getenv("HNSECW_DATA_FILE");
    if (!env || std::string(env).empty()) return;
    std::ifstream in(env, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[HNSECW_DATA_FILE] failed to open: " << env << std::endl;
        std::exit(1);
    }
    uint32_t header[4];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    g.N = header[0];
    g.D = header[1];
    g.M = header[2];
    g.num_queries = header[3];
    g.graph.resize((size_t)g.N * g.M);
    g.vec.resize((size_t)g.N * g.D);
    g.query.resize((size_t)g.num_queries * g.D);
    in.read(reinterpret_cast<char*>(g.graph.data()), g.graph.size() * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(g.vec.data()), g.vec.size() * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(g.query.data()), g.query.size() * sizeof(uint32_t));
    if (!in) {
        std::cerr << "[HNSECW_DATA_FILE] truncated file: " << env << std::endl;
        std::exit(1);
    }
    g.loaded = true;
    std::cerr << "[HNSECW_DATA_FILE] loaded N=" << g.N << " D=" << g.D << " M=" << g.M
              << " Q=" << g.num_queries << std::endl;
}

struct Layer {
    uint32_t N = 0;      // real nodes: addresses [0, N)
    uint32_t M = 0;
    uint32_t D = 0;
    uint32_t bitlen = 0;
    uint32_t sigma = 0;  // reserved address that is in the visited list from the start
    bool real = false;

    // Neighbour addresses of node u.  Unused slots and the rows of dummy nodes
    // hold sigma, which the visited check always rejects.
    std::vector<uint32_t> Neighbors(uint32_t u) const {
        std::vector<uint32_t> out(M, sigma);
        if (u >= N) return out;
        if (real) {
            const LayerData& g = Data();
            for (uint32_t k = 0; k < M; k++) {
                uint32_t v = g.graph[(size_t)u * M + k];
                out[k] = (v < N) ? v : sigma;
            }
            return out;
        }
        std::unordered_set<uint32_t> used;
        uint32_t state = u * 0x9e3779b9u + 0xa5a5a5a5u;
        uint32_t k = 0;
        const uint32_t want = (N > M) ? M : (N - 1);
        while (k < want) {
            state = state * 1664525u + 1013904223u;
            uint32_t v = (state >> 4) % N;
            if (v == u || !used.insert(v).second) continue;
            out[k++] = v;
        }
        return out;
    }

    std::vector<uint32_t> Vector(uint32_t id) const {
        std::vector<uint32_t> out(D, 0);
        if (id >= N) return out;
        if (real) {
            const LayerData& g = Data();
            for (uint32_t j = 0; j < D; j++) out[j] = g.vec[(size_t)id * D + j];
            return out;
        }
        const uint64_t bound = 1ULL << std::min<uint32_t>(bitlen, 31);
        for (uint32_t j = 0; j < D; j++) {
            uint64_t v = (uint64_t)id * 2654435761u + (uint64_t)j * 2246822519u;
            out[j] = (uint32_t)((v >> 7) % bound);
        }
        return out;
    }

    // Query number qi: taken from the data file, or synthetic.
    std::vector<uint32_t> Query(uint32_t qi) const {
        std::vector<uint32_t> q(D);
        const LayerData& g = Data();
        if (real && g.num_queries > 0) {
            if (qi >= g.num_queries) {
                std::cerr << "query " << qi << " requested but the data file holds "
                          << g.num_queries << std::endl;
                std::exit(1);
            }
            for (uint32_t j = 0; j < D; j++) q[j] = g.query[(size_t)qi * D + j];
            return q;
        }
        uint32_t s = 12345u + 7919u * qi;
        const uint32_t bound = 1U << std::min<uint32_t>(bitlen, 31);
        for (uint32_t j = 0; j < D; j++) {
            s = s * 1664525u + 1013904223u;
            q[j] = (s >> 8) % bound;
        }
        return q;
    }
};

// Sets up the layer and its address space:
//   [0, N)            real nodes
//   N + i, i < pool   dummy addresses handed out by the caller
//   N + pool          sigma
// Returns false (after printing why) if id_bitlen is too small.
inline bool InitLayer(Layer& layer, uint32_t M, uint32_t D, uint32_t bitlen,
                      uint32_t id_bitlen, uint64_t pool) {
    LoadLayerData();
    const LayerData& g = Data();
    layer.M = M;
    layer.D = D;
    layer.bitlen = bitlen;
    layer.real = g.loaded;
    const uint64_t space = 1ULL << id_bitlen;
    if (g.loaded && (g.M != M || g.D != D)) {
        std::cerr << "[HNSECW_DATA_FILE] shape mismatch: file has M=" << g.M << " D=" << g.D
                  << ", command line has M=" << M << " D=" << D << std::endl;
        return false;
    }
    if (layer.real) {
        layer.N = g.N;
        if ((uint64_t)layer.N + pool + 1 > space) {
            std::cerr << "id_bitlen=" << id_bitlen << " too small for N=" << layer.N
                      << " plus " << (pool + 1) << " dummy addresses" << std::endl;
            return false;
        }
    } else {
        if (pool + 3 > space) {
            std::cerr << "id_bitlen=" << id_bitlen << " too small for " << (pool + 1)
                      << " dummy addresses" << std::endl;
            return false;
        }
        layer.N = (uint32_t)(space - pool - 1);
    }
    layer.sigma = layer.N + (uint32_t)pool;
    return true;
}

// ---------------------------------------------------------------------------
// Cost accounting
// ---------------------------------------------------------------------------

struct Cost {
    double online_ms = 0.0;
    double setup_ms = 0.0;
    uint64_t online_bytes = 0;
    uint64_t setup_bytes = 0;
    uint32_t rounds = 0;  // online interaction layers

    void Add(const Cost& o) {
        online_ms += o.online_ms;
        setup_ms += o.setup_ms;
        online_bytes += o.online_bytes;
        setup_bytes += o.setup_bytes;
        rounds += o.rounds;
    }
};

inline Cost Execute(ABYParty* party) {
    party->ExecCircuit();
    Cost c;
    c.online_ms = party->GetTiming(P_ONLINE);
    c.setup_ms = party->GetTiming(P_SETUP);
    c.online_bytes = party->GetSentData(P_ONLINE) + party->GetReceivedData(P_ONLINE);
    c.setup_bytes = party->GetSentData(P_SETUP) + party->GetReceivedData(P_SETUP);
    for (Sharing* sh : party->GetSharings()) {
        c.rounds = std::max(c.rounds, sh->GetMaxCommunicationRounds());
    }
    return c;
}

inline void PrintCost(const std::string& label, const Cost& c) {
    const double mb = 1024.0 * 1024.0;
    std::cout << label << " online latency(s)=" << (c.online_ms / 1000.0)
              << " comm(MB)=" << (c.online_bytes / mb) << " rounds=" << c.rounds << std::endl;
}

inline void PrintTotals(const Cost& total) {
    const double mb = 1024.0 * 1024.0;
    std::cout << "[Total Setup] latency(s)=" << (total.setup_ms / 1000.0)
              << " comm(MB)=" << (total.setup_bytes / mb) << std::endl;
    std::cout << "[Total Online] latency(s)=" << (total.online_ms / 1000.0)
              << " comm(MB)=" << (total.online_bytes / mb)
              << " rounds=" << total.rounds << std::endl;
}

// ---------------------------------------------------------------------------
// Circuit helpers
// ---------------------------------------------------------------------------

inline std::vector<uint32_t> ReadSimd(share* out, uint32_t expect_nvals) {
    uint32_t* vals = nullptr;
    uint32_t bits = 0;
    uint32_t nvals = 0;
    out->get_clear_value_vec(&vals, &bits, &nvals);
    if (nvals != expect_nvals) {
        std::cerr << "output gate returned " << nvals << " values, expected "
                  << expect_nvals << std::endl;
        std::exit(1);
    }
    std::vector<uint32_t> v(vals, vals + nvals);
    free(vals);
    return v;
}

struct ShareBin {
    std::vector<share*> items;
    share* Keep(share* s) {
        items.push_back(s);
        return s;
    }
    ~ShareBin() {
        for (share* s : items) delete s;
    }
};

inline std::vector<uint32_t> Range(uint32_t lo, uint32_t hi) {
    std::vector<uint32_t> r;
    r.reserve(hi > lo ? hi - lo : 0);
    for (uint32_t i = lo; i < hi; i++) r.push_back(i);
    return r;
}

// Opens Yao wires to both parties.  The value goes through a (free) Y2B
// conversion first: ABY matches Yao output gates between garbler and
// evaluator by creation order on one side and by circuit depth on the other,
// which only agree for a single output gate.
inline share* OpenYao(BooleanCircuit* bc, BooleanCircuit* yc, ShareBin& bin,
                      const std::vector<uint32_t>& wires) {
    share* y = bin.Keep(new boolshare(wires, yc));
    share* b = bin.Keep(bc->PutY2BGate(y));
    return bin.Keep(bc->PutOUTGate(b, ALL));
}

inline share* YaoToSharedOut(BooleanCircuit* bc, BooleanCircuit* yc, ShareBin& bin,
                             const std::vector<uint32_t>& wires) {
    share* y = bin.Keep(new boolshare(wires, yc));
    share* b = bin.Keep(bc->PutY2BGate(y));
    return bin.Keep(bc->PutSharedOUTGate(b));
}

// XOR shares of n values of `bits` bits -> Yao wires (nvals = n per wire).
inline std::vector<uint32_t> SharedToYaoWires(BooleanCircuit* bc, BooleanCircuit* yc, ShareBin& bin,
                                              std::vector<uint32_t>& shares, uint32_t bits) {
    share* b = bin.Keep(bc->PutSharedSIMDINGate((uint32_t)shares.size(), shares.data(), bits));
    share* y = bin.Keep(yc->PutB2YGate(b));
    return y->get_wires();
}

// Repeats a wire `times` times back to back.
inline uint32_t Tile(BooleanCircuit* c, uint32_t wire, uint32_t times) {
    if (times == 1) return wire;
    return c->PutCombinerGate(std::vector<uint32_t>(times, wire));
}

}  // namespace hnsecw

#endif  // HNSECW_HARNESS_H
