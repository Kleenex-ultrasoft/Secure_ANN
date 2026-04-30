#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>
#include <cryptoTools/Network/Session.h>

#include "aby3/Circuit/CircuitLibrary.h"
#include "aby3/sh3/Sh3BinaryEvaluator.h"
#include "aby3/sh3/Sh3Converter.h"
#include "aby3/sh3/Sh3Encryptor.h"
#include "aby3/sh3/Sh3Evaluator.h"
#include "aby3/sh3/Sh3Runtime.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
using namespace oc;
using namespace aby3;

enum class PerfEvent {
    LessThan = 0,
    SelectRow = 1,
    SelectValue = 2,
    DotProduct = 3,
    BitonicSort = 4,
    BatchDedup = 5,
    CacheLookup = 6,
    Membership = 7,
    Count = 8,
};

struct PerfStats {
    bool enabled = false;
    std::array<double, static_cast<size_t>(PerfEvent::Count)> seconds{};
    std::array<u64, static_cast<size_t>(PerfEvent::Count)> calls{};
    std::array<u64, static_cast<size_t>(PerfEvent::Count)> items{};

    void reset() {
        seconds.fill(0.0);
        calls.fill(0);
        items.fill(0);
    }

    void add(PerfEvent event, double sec, u64 items_add) {
        size_t idx = static_cast<size_t>(event);
        seconds[idx] += sec;
        calls[idx] += 1;
        items[idx] += items_add;
    }
};

static PerfStats g_perf;

static bool aby3_debug_enabled() {
    const char* env = std::getenv("HNSECW_ABY3_DEBUG");
    if (!env || !*env) {
        return false;
    }
    return std::string(env) != "0";
}

static const char* perf_name(PerfEvent event) {
    switch (event) {
    case PerfEvent::LessThan:
        return "less_than";
    case PerfEvent::SelectRow:
        return "select_row";
    case PerfEvent::SelectValue:
        return "select_value";
    case PerfEvent::DotProduct:
        return "dot_product";
    case PerfEvent::BitonicSort:
        return "bitonic_sort";
    case PerfEvent::BatchDedup:
        return "batch_dedup";
    case PerfEvent::CacheLookup:
        return "cache_lookup";
    case PerfEvent::Membership:
        return "membership";
    case PerfEvent::Count:
    default:
        return "unknown";
    }
}

static void perf_reset() {
    g_perf.reset();
}

static void perf_print(const std::string& tag, u64 party) {
    if (!g_perf.enabled || party != 0) {
        return;
    }
    std::cout << "[perf] " << tag << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < static_cast<size_t>(PerfEvent::Count); ++i) {
        if (g_perf.calls[i] == 0) {
            continue;
        }
        double total_ms = g_perf.seconds[i] * 1000.0;
        double avg_ms = total_ms / static_cast<double>(g_perf.calls[i]);
        std::cout << "[perf] " << perf_name(static_cast<PerfEvent>(i))
                  << " total_ms=" << total_ms
                  << " avg_ms=" << avg_ms
                  << " calls=" << g_perf.calls[i]
                  << " items=" << g_perf.items[i]
                  << std::endl;
    }
}

class ScopedTimer {
public:
    using Clock = std::chrono::steady_clock;

    ScopedTimer(PerfEvent event, u64 items = 0)
        : event_(event),
          items_(items),
          active_(g_perf.enabled) {
        if (active_) {
            start_ = Clock::now();
        }
    }

    ~ScopedTimer() {
        if (!active_) {
            return;
        }
        double sec = std::chrono::duration<double>(Clock::now() - start_).count();
        g_perf.add(event_, sec, items_);
    }

private:
    PerfEvent event_;
    u64 items_;
    bool active_;
    Clock::time_point start_{};
};

struct MpcContext {
    u64 party = 0;
    IOService ios;
    Sh3Runtime rt;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Converter conv;
};

struct CommStats {
    u64 bytes_sent = 0;
    u64 bytes_recv = 0;

    u64 total() const { return bytes_sent + bytes_recv; }
    u64 sent() const { return bytes_sent; }
    u64 recv() const { return bytes_recv; }
};

static CommStats get_comm_stats(MpcContext& ctx) {
    CommStats stats;
    stats.bytes_sent = ctx.rt.mComm.mNext.getTotalDataSent()
                     + ctx.rt.mComm.mPrev.getTotalDataSent();
    stats.bytes_recv = ctx.rt.mComm.mNext.getTotalDataRecv()
                     + ctx.rt.mComm.mPrev.getTotalDataRecv();
    return stats;
}

struct LayerConfig {
    u64 idx = 0;
    u64 n = 0;
    u64 n_real = 0;
    u64 m = 0;
    u64 t = 0;
    u64 l_c = 0;
    u64 l_w = 0;
    u64 id_bits = 0;
    u64 vec_bits = 0;
    u64 x2_bits = 0;
    u64 down_bits = 0;
};

struct LayerData {
    LayerConfig cfg;
    si64Matrix graph;
    si64Matrix vecs;
    std::vector<si64> down;
    std::vector<si64> is_dummy;
    si64 dummy_id;
    u64 dummy_open = 0;
};

struct BatchDedupResult {
    std::vector<si64> ids;
    std::vector<si64> hits;
};

static u64 ceil_log2_u64(u64 n) {
    if (n <= 1) {
        return 1;
    }
    return static_cast<u64>(std::ceil(std::log2(static_cast<double>(n))));
}

static u64 next_pow2(u64 n) {
    if (n <= 1) {
        return 1;
    }
    u64 p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

static u64 dist_bits_for_layer(u64 vec_bits, u64 dim) {
    return 2 * vec_bits + ceil_log2_u64(dim) + 1;
}

static si64 make_public_share(i64 value, u64 party) {
    si64 out;
    if (party == 0) {
        out[0] = value;
        out[1] = 0;
    } else if (party == 1) {
        out[0] = 0;
        out[1] = value;
    } else {
        out[0] = 0;
        out[1] = 0;
    }
    return out;
}

static si64 mul_const(const si64& x, i64 c) {
    si64 out;
    out[0] = x[0] * c;
    out[1] = x[1] * c;
    return out;
}

static size_t byte_width(u64 bitlen) {
    if (bitlen <= 8) {
        return 1;
    }
    if (bitlen <= 16) {
        return 2;
    }
    if (bitlen <= 32) {
        return 4;
    }
    return 8;
}

template <typename T>
static std::vector<i64> read_bin_t(const std::string& path, size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open input: " + path);
    }
    std::vector<T> buf(count);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw std::runtime_error("failed to read input: " + path);
    }
    std::vector<i64> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<i64>(buf[i]);
    }
    return out;
}

static std::vector<i64> read_bin(const std::string& path, size_t count, u64 bitlen) {
    switch (byte_width(bitlen)) {
    case 1:
        return read_bin_t<uint8_t>(path, count);
    case 2:
        return read_bin_t<uint16_t>(path, count);
    case 4:
        return read_bin_t<uint32_t>(path, count);
    default:
        return read_bin_t<uint64_t>(path, count);
    }
}

static size_t file_entry_count(const std::string& path, u64 bitlen) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open input: " + path);
    }
    std::streamsize size = in.tellg();
    size_t width = byte_width(bitlen);
    if (size < 0 || (static_cast<size_t>(size) % width) != 0) {
        throw std::runtime_error("invalid binary length for " + path);
    }
    return static_cast<size_t>(size) / width;
}

template <typename T>
static void write_bin_t(const std::string& path, const std::vector<i64>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output: " + path);
    }
    std::vector<T> buf(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        buf[i] = static_cast<T>(values[i]);
    }
    out.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(buf.size() * sizeof(T)));
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

static void write_bin(const std::string& path, const std::vector<i64>& values, u64 bitlen) {
    switch (byte_width(bitlen)) {
    case 1:
        write_bin_t<uint8_t>(path, values);
        break;
    case 2:
        write_bin_t<uint16_t>(path, values);
        break;
    case 4:
        write_bin_t<uint32_t>(path, values);
        break;
    default:
        write_bin_t<uint64_t>(path, values);
        break;
    }
}

static std::vector<si64> load_share_vector(
    const std::string& s0_path,
    const std::string& s1_path,
    size_t count,
    u64 bitlen) {
    auto s0 = read_bin(s0_path, count, bitlen);
    auto s1 = read_bin(s1_path, count, bitlen);
    if (s0.size() != s1.size()) {
        throw std::runtime_error("share size mismatch for " + s0_path);
    }
    std::vector<si64> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i][0] = s0[i];
        out[i][1] = s1[i];
    }
    return out;
}

static si64Matrix load_share_matrix(
    const std::string& s0_path,
    const std::string& s1_path,
    size_t rows,
    size_t cols,
    u64 bitlen) {
    auto s0 = read_bin(s0_path, rows * cols, bitlen);
    auto s1 = read_bin(s1_path, rows * cols, bitlen);
    if (s0.size() != s1.size()) {
        throw std::runtime_error("share size mismatch for " + s0_path);
    }
    si64Matrix mat(rows, cols);
    size_t idx = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j, ++idx) {
            mat.mShares[0](i, j) = s0[idx];
            mat.mShares[1](i, j) = s1[idx];
        }
    }
    return mat;
}

static si64Matrix vector_to_matrix(const std::vector<si64>& vec, size_t rows, size_t cols) {
    si64Matrix mat(rows, cols);
    size_t idx = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j, ++idx) {
            mat.mShares[0](i, j) = vec[idx][0];
            mat.mShares[1](i, j) = vec[idx][1];
        }
    }
    return mat;
}

static std::vector<si64> matrix_to_vector(const si64Matrix& mat) {
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    std::vector<si64> out(rows * cols);
    size_t idx = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j, ++idx) {
            out[idx][0] = mat.mShares[0](i, j);
            out[idx][1] = mat.mShares[1](i, j);
        }
    }
    return out;
}

static inline si64 matrix_get(const si64Matrix& mat, size_t row, size_t col) {
    si64 out;
    out[0] = mat.mShares[0](row, col);
    out[1] = mat.mShares[1](row, col);
    return out;
}

// Avoid sMatrix::operator() assignment because Ref<Share>::operator= does not copy values.
static inline void matrix_set(si64Matrix& mat, size_t row, size_t col, const si64& val) {
    mat.mShares[0](row, col) = val[0];
    mat.mShares[1](row, col) = val[1];
}

class ComparatorCache {
public:
    explicit ComparatorCache(MpcContext& ctx) : ctx_(ctx) {}

    std::vector<si64> less_than(const std::vector<si64>& a,
                                const std::vector<si64>& b,
                                u64 bitlen) {
        ScopedTimer timer(PerfEvent::LessThan, a.size());
        if (a.size() != b.size()) {
            throw std::runtime_error("compare_lt size mismatch");
        }
        size_t n = a.size();
        if (n == 0) {
            return {};
        }
        si64Matrix a_mat = vector_to_matrix(a, n, 1);
        si64Matrix b_mat = vector_to_matrix(b, n, 1);
        sbMatrix a_bin(n, bitlen);
        sbMatrix b_bin(n, bitlen);

        Sh3Task t = ctx_.rt.noDependencies();
        t &= ctx_.conv.toBinaryMatrix(t, a_mat, a_bin);
        t &= ctx_.conv.toBinaryMatrix(t, b_mat, b_bin);
        t.get();

        oc::BetaCircuit* cir = lt_circuit(bitlen);
        Sh3BinaryEvaluator eval;
        eval.setCir(cir, n, ctx_.enc.mShareGen);
        eval.setInput(0, a_bin);
        eval.setInput(1, b_bin);
        eval.asyncEvaluate(ctx_.rt.noDependencies()).get();

        sbMatrix out_bin(n, 1);
        eval.getOutput(0, out_bin);

        si64Matrix out_mat(n, 1);
        ctx_.conv.bitInjection(ctx_.rt.noDependencies(), out_bin, out_mat, false).get();

        return matrix_to_vector(out_mat);
    }

    std::vector<si64> equal(const std::vector<si64>& a,
                             const std::vector<si64>& b,
                             u64 bitlen) {
        ScopedTimer timer(PerfEvent::LessThan, a.size());
        if (a.size() != b.size()) {
            throw std::runtime_error("compare_eq size mismatch");
        }
        size_t n = a.size();
        if (n == 0) {
            return {};
        }
        si64Matrix a_mat = vector_to_matrix(a, n, 1);
        si64Matrix b_mat = vector_to_matrix(b, n, 1);
        sbMatrix a_bin(n, bitlen);
        sbMatrix b_bin(n, bitlen);

        Sh3Task t = ctx_.rt.noDependencies();
        t &= ctx_.conv.toBinaryMatrix(t, a_mat, a_bin);
        t &= ctx_.conv.toBinaryMatrix(t, b_mat, b_bin);
        t.get();

        oc::BetaCircuit* cir = eq_circuit(bitlen);
        Sh3BinaryEvaluator eval;
        eval.setCir(cir, n, ctx_.enc.mShareGen);
        eval.setInput(0, a_bin);
        eval.setInput(1, b_bin);
        eval.asyncEvaluate(ctx_.rt.noDependencies()).get();

        sbMatrix out_bin(n, 1);
        eval.getOutput(0, out_bin);

        si64Matrix out_mat(n, 1);
        ctx_.conv.bitInjection(ctx_.rt.noDependencies(), out_bin, out_mat, false).get();

        return matrix_to_vector(out_mat);
    }

private:
    oc::BetaCircuit* lt_circuit(u64 bitlen) {
        auto it = lt_circuits_.find(bitlen);
        if (it != lt_circuits_.end()) {
            return it->second.get();
        }
        auto cir = std::make_unique<oc::BetaCircuit>();
        oc::BetaBundle a(bitlen);
        oc::BetaBundle b(bitlen);
        oc::BetaBundle out(1);
        cir->addInputBundle(a);
        cir->addInputBundle(b);
        cir->addOutputBundle(out);
        lib_.lessThan_build(*cir, a, b, out,
                            oc::BetaLibrary::IntType::TwosComplement,
                            oc::BetaLibrary::Optimized::Size);
        auto* ptr = cir.get();
        lt_circuits_[bitlen] = std::move(cir);
        return ptr;
    }

    oc::BetaCircuit* eq_circuit(u64 bitlen) {
        auto it = eq_circuits_.find(bitlen);
        if (it != eq_circuits_.end()) {
            return it->second.get();
        }
        auto cir = std::make_unique<oc::BetaCircuit>();
        oc::BetaBundle a(bitlen);
        oc::BetaBundle b(bitlen);
        oc::BetaBundle out(1);
        cir->addInputBundle(a);
        cir->addInputBundle(b);
        cir->addOutputBundle(out);
        lib_.eq_build(*cir, a, b, out);
        auto* ptr = cir.get();
        eq_circuits_[bitlen] = std::move(cir);
        return ptr;
    }

    MpcContext& ctx_;
    aby3::CircuitLibrary lib_;
    std::unordered_map<u64, std::unique_ptr<oc::BetaCircuit>> lt_circuits_;
    std::unordered_map<u64, std::unique_ptr<oc::BetaCircuit>> eq_circuits_;
};

static si64 mul_scalar(MpcContext& ctx, const si64& a, const si64& b) {
    si64Matrix a_mat(1, 1);
    si64Matrix b_mat(1, 1);
    matrix_set(a_mat, 0, 0, a);
    matrix_set(b_mat, 0, 0, b);
    si64Matrix out(1, 1);
    ctx.eval.asyncMul(ctx.rt.noDependencies(), a_mat, b_mat, out).get();
    return matrix_get(out, 0, 0);
}

static std::vector<si64> mul_vec(MpcContext& ctx,
                                 const std::vector<si64>& a,
                                 const std::vector<si64>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("mul_vec size mismatch");
    }
    size_t n = a.size();
    if (n == 0) {
        return {};
    }
    si64Matrix a_mat(n, 1);
    si64Matrix b_mat(n, 1);
    for (size_t i = 0; i < n; ++i) {
        matrix_set(a_mat, i, 0, a[i]);
        matrix_set(b_mat, i, 0, b[i]);
    }
    si64Matrix out_mat(n, 1);
    ctx.eval.asyncMul(ctx.rt.noDependencies(), a_mat, b_mat, out_mat).get();
    std::vector<si64> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = matrix_get(out_mat, i, 0);
    }
    return out;
}

static std::vector<si64> select_vec_by_bit(MpcContext& ctx,
                                           const std::vector<si64>& base,
                                           const std::vector<si64>& bit,
                                           const si64& alt) {
    if (base.size() != bit.size()) {
        throw std::runtime_error("select_vec_by_bit size mismatch");
    }
    std::vector<si64> delta(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        delta[i] = alt - base[i];
    }
    auto prod = mul_vec(ctx, bit, delta);
    std::vector<si64> out(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        out[i] = base[i] + prod[i];
    }
    return out;
}

static si64 sum_vec(const std::vector<si64>& vec, u64 party) {
    si64 acc = make_public_share(0, party);
    for (const auto& v : vec) {
        acc = acc + v;
    }
    return acc;
}

static u64 mask_for_bits(u64 bits) {
    if (bits >= 64) {
        return ~0ULL;
    }
    if (bits == 0) {
        return 0;
    }
    return (1ULL << bits) - 1ULL;
}

static u64 reveal_id(MpcContext& ctx, const si64& id, u64 id_bits) {
    i64 plain = 0;
    ctx.enc.revealAll(ctx.rt.noDependencies(), id, plain).get();
    u64 mask = mask_for_bits(id_bits);
    return static_cast<u64>(plain) & mask;
}

static std::vector<u64> reveal_ids(MpcContext& ctx, const std::vector<si64>& ids, u64 id_bits) {
    if (ids.empty()) {
        return {};
    }
    si64Matrix mat = vector_to_matrix(ids, ids.size(), 1);
    i64Matrix plain(ids.size(), 1);
    ctx.enc.revealAll(ctx.rt.noDependencies(), mat, plain).get();
    u64 mask = mask_for_bits(id_bits);
    std::vector<u64> out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        out[i] = static_cast<u64>(plain(i, 0)) & mask;
    }
    return out;
}

static u64 normalize_id(u64 id, const LayerData& layer) {
    u64 real = layer.cfg.n_real ? layer.cfg.n_real : layer.cfg.n;
    if (id >= real) {
        return layer.dummy_open;
    }
    if (id >= layer.cfg.n) {
        return layer.dummy_open;
    }
    return id;
}

static std::vector<si64> fetch_row_public(const si64Matrix& mat, u64 row, u64 fallback_row) {
    u64 rows = static_cast<u64>(mat.rows());
    if (rows == 0) {
        throw std::runtime_error("public row index on empty matrix");
    }
    if (row >= rows) {
        row = fallback_row;
    }
    if (row >= rows) {
        row = rows - 1;
    }
    size_t cols = mat.cols();
    std::vector<si64> out(cols);
    for (size_t j = 0; j < cols; ++j) {
        out[j] = mat(row, j);
    }
    return out;
}

static si64 fetch_value_public(const std::vector<si64>& vec, u64 idx, u64 fallback_idx) {
    if (vec.empty()) {
        throw std::runtime_error("public index on empty vector");
    }
    if (idx >= vec.size()) {
        idx = fallback_idx;
    }
    if (idx >= vec.size()) {
        idx = vec.size() - 1;
    }
    return vec[idx];
}

static bool lookup_graph_cache_plain(const std::vector<u64>& keys,
                                     const si64Matrix& vals,
                                     size_t hist_len,
                                     u64 key,
                                     std::vector<si64>& out) {
    bool hit = false;
    if (hist_len == 0) {
        return false;
    }
    if (hist_len > keys.size()) {
        throw std::runtime_error("graph cache hist_len exceeds key size");
    }
    out.assign(vals.cols(), si64{});
    for (size_t i = 0; i < hist_len; ++i) {
        if (keys[i] == key) {
            hit = true;
            for (size_t j = 0; j < static_cast<size_t>(vals.cols()); ++j) {
                out[j] = vals(i, j);
            }
        }
    }
    return hit;
}

static bool lookup_vector_cache_plain(const std::vector<u64>& keys,
                                      const si64Matrix& vals,
                                      size_t hist_len,
                                      u64 key,
                                      std::vector<si64>& out) {
    bool hit = false;
    if (hist_len == 0) {
        return false;
    }
    if (hist_len > keys.size()) {
        throw std::runtime_error("vector cache hist_len exceeds key size");
    }
    out.assign(vals.cols(), si64{});
    for (size_t i = 0; i < hist_len; ++i) {
        if (keys[i] == key) {
            hit = true;
            for (size_t j = 0; j < static_cast<size_t>(vals.cols()); ++j) {
                out[j] = vals(i, j);
            }
        }
    }
    return hit;
}

static si64 or_reduce_tree(MpcContext& ctx, std::vector<si64>& bits) {
    if (bits.empty()) {
        return make_public_share(0, ctx.party);
    }
    while (bits.size() > 1) {
        size_t half = bits.size() / 2;
        std::vector<si64> left(half);
        std::vector<si64> right(half);
        for (size_t i = 0; i < half; ++i) {
            left[i] = bits[2 * i];
            right[i] = bits[2 * i + 1];
        }
        auto prod = mul_vec(ctx, left, right);
        std::vector<si64> next(half + (bits.size() % 2));
        for (size_t i = 0; i < half; ++i) {
            next[i] = left[i] + right[i] - prod[i];
        }
        if (bits.size() % 2 == 1) {
            next[half] = bits.back();
        }
        bits = std::move(next);
    }
    return bits[0];
}

static si64 membership(MpcContext& ctx,
                       ComparatorCache& cmp,
                       const si64& key,
                       const std::vector<si64>& ids,
                       const std::vector<si64>& valid,
                       u64 id_bits) {
    ScopedTimer timer(PerfEvent::Membership, ids.size());
    if (ids.empty()) {
        return make_public_share(0, ctx.party);
    }
    std::vector<si64> keys(ids.size(), key);
    auto eq = cmp.equal(keys, ids, id_bits);
    auto match = mul_vec(ctx, eq, valid);
    return or_reduce_tree(ctx, match);
}

static std::vector<si64> membership_batch(MpcContext& ctx,
                                          ComparatorCache& cmp,
                                          const std::vector<si64>& keys,
                                          const std::vector<si64>& ids,
                                          const std::vector<si64>& valid,
                                          u64 id_bits) {
    ScopedTimer timer(PerfEvent::Membership, keys.size() * ids.size());
    size_t k = keys.size();
    size_t v = ids.size();
    if (k == 0) {
        return {};
    }
    if (v == 0) {
        std::vector<si64> out(k);
        for (size_t i = 0; i < k; ++i) {
            out[i] = make_public_share(0, ctx.party);
        }
        return out;
    }

    std::vector<si64> all_keys(k * v);
    std::vector<si64> all_ids(k * v);
    std::vector<si64> all_valid(k * v);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < v; ++j) {
            size_t idx = i * v + j;
            all_keys[idx] = keys[i];
            all_ids[idx] = ids[j];
            all_valid[idx] = valid[j];
        }
    }

    auto eq = cmp.equal(all_keys, all_ids, id_bits);
    auto match = mul_vec(ctx, eq, all_valid);

    std::vector<si64> out(k);
    for (size_t i = 0; i < k; ++i) {
        std::vector<si64> row(v);
        for (size_t j = 0; j < v; ++j) {
            row[j] = match[i * v + j];
        }
        out[i] = or_reduce_tree(ctx, row);
    }
    return out;
}

static void apply_swaps_multi(MpcContext& ctx,
                              const std::vector<size_t>& a_idx,
                              const std::vector<size_t>& b_idx,
                              const std::vector<si64>& swap_bits,
                              const std::vector<std::vector<si64>*>& arrays) {
    if (a_idx.empty()) {
        return;
    }
    size_t n = a_idx.size();

    for (auto* arr : arrays) {
        std::vector<si64> diff(n);
        for (size_t i = 0; i < n; ++i) {
            diff[i] = (*arr)[b_idx[i]] - (*arr)[a_idx[i]];
        }
        auto prod = mul_vec(ctx, diff, swap_bits);
        for (size_t i = 0; i < n; ++i) {
            (*arr)[a_idx[i]] = (*arr)[a_idx[i]] + prod[i];
            (*arr)[b_idx[i]] = (*arr)[b_idx[i]] - prod[i];
        }
    }
}

static void bitonic_sort_pairs(MpcContext& ctx,
                               ComparatorCache& cmp,
                               std::vector<si64>& keys,
                               const std::vector<std::vector<si64>*>& payloads,
                               u64 key_bits) {
    ScopedTimer timer(PerfEvent::BitonicSort, keys.size());
    size_t n = keys.size();
    if (n <= 1) {
        return;
    }
    for (size_t k = 2; k <= n; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            std::vector<size_t> a_up;
            std::vector<size_t> b_up;
            std::vector<size_t> a_down;
            std::vector<size_t> b_down;
            std::vector<si64> a_up_vals;
            std::vector<si64> b_up_vals;
            std::vector<si64> a_down_vals;
            std::vector<si64> b_down_vals;

            for (size_t i = 0; i < n; ++i) {
                size_t ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    if (up) {
                        a_up.push_back(i);
                        b_up.push_back(ixj);
                        a_up_vals.push_back(keys[i]);
                        b_up_vals.push_back(keys[ixj]);
                    } else {
                        a_down.push_back(i);
                        b_down.push_back(ixj);
                        a_down_vals.push_back(keys[i]);
                        b_down_vals.push_back(keys[ixj]);
                    }
                }
            }

            std::vector<std::vector<si64>*> arrays;
            arrays.reserve(1 + payloads.size());
            arrays.push_back(&keys);
            arrays.insert(arrays.end(), payloads.begin(), payloads.end());

            if (!a_up.empty()) {
                auto swap_up = cmp.less_than(b_up_vals, a_up_vals, key_bits);
                apply_swaps_multi(ctx, a_up, b_up, swap_up, arrays);
            }
            if (!a_down.empty()) {
                auto swap_down = cmp.less_than(a_down_vals, b_down_vals, key_bits);
                apply_swaps_multi(ctx, a_down, b_down, swap_down, arrays);
            }
        }
    }
}

static void sort_pairs(MpcContext& ctx,
                       ComparatorCache& cmp,
                       std::vector<si64>& dist,
                       std::vector<si64>& ids,
                       u64 key_bits,
                       const si64& pad_key,
                       const si64& pad_id) {
    size_t n = dist.size();
    size_t pow2 = next_pow2(n);
    while (dist.size() < pow2) {
        dist.push_back(pad_key);
        ids.push_back(pad_id);
    }
    bitonic_sort_pairs(ctx, cmp, dist, {&ids}, key_bits);
    dist.resize(n);
    ids.resize(n);
}

static void merge_topk(MpcContext& ctx,
                       ComparatorCache& cmp,
                       std::vector<si64>& cur_dist,
                       std::vector<si64>& cur_id,
                       const std::vector<si64>& cand_dist,
                       const std::vector<si64>& cand_id,
                       size_t keep,
                       u64 key_bits,
                       const si64& pad_key,
                       const si64& pad_id) {
    std::vector<si64> keys = cur_dist;
    std::vector<si64> ids = cur_id;
    keys.insert(keys.end(), cand_dist.begin(), cand_dist.end());
    ids.insert(ids.end(), cand_id.begin(), cand_id.end());

    size_t n = keys.size();
    size_t pow2 = next_pow2(n);
    while (keys.size() < pow2) {
        keys.push_back(pad_key);
        ids.push_back(pad_id);
    }

    bitonic_sort_pairs(ctx, cmp, keys, {&ids}, key_bits);
    keys.resize(keep);
    ids.resize(keep);
    cur_dist = keys;
    cur_id = ids;
}

static BatchDedupResult batch_dedup(MpcContext& ctx,
                                    ComparatorCache& cmp,
                                    const std::vector<si64>& cand_ids,
                                    const std::vector<si64>& dummy_ids,
                                    const std::vector<u64>& visited_open,
                                    u64 id_bits) {
    ScopedTimer timer(PerfEvent::BatchDedup, cand_ids.size());
    size_t k = cand_ids.size();
    size_t v = visited_open.size();
    size_t total = v + k;

    si64 sentinel = make_public_share(static_cast<i64>(1ULL << id_bits), ctx.party);
    std::vector<si64> keys(total);
    std::vector<si64> tags(total);
    std::vector<si64> ids(total);
    std::vector<si64> dummies(total);
    std::vector<si64> idxs(total);

    si64 zero = make_public_share(0, ctx.party);
    si64 one = make_public_share(1, ctx.party);
    for (size_t i = 0; i < v; ++i) {
        si64 vid = make_public_share(static_cast<i64>(visited_open[i]), ctx.party);
        keys[i] = vid + vid;
        tags[i] = zero;
        ids[i] = vid;
        dummies[i] = zero;
        idxs[i] = make_public_share(static_cast<i64>(k), ctx.party);
    }
    for (size_t i = 0; i < k; ++i) {
        size_t row = v + i;
        keys[row] = cand_ids[i] + cand_ids[i] + one;
        tags[row] = one;
        ids[row] = cand_ids[i];
        dummies[row] = dummy_ids[i];
        idxs[row] = make_public_share(static_cast<i64>(i), ctx.party);
    }

    size_t pad = next_pow2(total);
    while (keys.size() < pad) {
        keys.push_back(sentinel + sentinel + one);
        tags.push_back(zero);
        ids.push_back(sentinel);
        dummies.push_back(zero);
        idxs.push_back(make_public_share(static_cast<i64>(k), ctx.party));
    }

    bitonic_sort_pairs(ctx, cmp, keys, {&tags, &ids, &dummies, &idxs}, id_bits + 2);
    keys.resize(total);
    tags.resize(total);
    ids.resize(total);
    dummies.resize(total);
    idxs.resize(total);

    std::vector<si64> prev_ids(total);
    prev_ids[0] = sentinel;
    for (size_t i = 1; i < total; ++i) {
        prev_ids[i] = ids[i - 1];
    }
    auto eq_prev_all = cmp.equal(ids, prev_ids, id_bits + 1);

    auto hits = mul_vec(ctx, tags, eq_prev_all);

    std::vector<si64> deltas(total);
    for (size_t i = 0; i < total; ++i) {
        deltas[i] = dummies[i] - ids[i];
    }
    auto adjs = mul_vec(ctx, hits, deltas);
    std::vector<si64> id_sel(total);
    for (size_t i = 0; i < total; ++i) {
        id_sel[i] = ids[i] + adjs[i];
    }

    std::vector<si64> idx_deltas(total);
    si64 k_val = make_public_share(static_cast<i64>(k), ctx.party);
    for (size_t i = 0; i < total; ++i) {
        idx_deltas[i] = idxs[i] - k_val;
    }
    auto key_adjs = mul_vec(ctx, tags, idx_deltas);

    std::vector<si64> out_key(total);
    std::vector<si64> out_id(total);
    std::vector<si64> out_hit(total);
    for (size_t i = 0; i < total; ++i) {
        out_key[i] = k_val + key_adjs[i];
        out_id[i] = id_sel[i];
        out_hit[i] = hits[i];
    }

    size_t out_pad = next_pow2(total);
    while (out_key.size() < out_pad) {
        out_key.push_back(make_public_share(static_cast<i64>(k + 1), ctx.party));
        out_id.push_back(sentinel);
        out_hit.push_back(zero);
    }

    u64 row_bits = ceil_log2_u64(k + 2);
    bitonic_sort_pairs(ctx, cmp, out_key, {&out_id, &out_hit}, row_bits);

    BatchDedupResult res;
    res.ids.resize(k);
    res.hits.resize(k);
    for (size_t i = 0; i < k; ++i) {
        res.ids[i] = out_id[i];
        res.hits[i] = out_hit[i];
    }
    return res;
}

static si64 dot_product(MpcContext& ctx,
                        const std::vector<si64>& vec,
                        const std::vector<si64>& query) {
    // Despite the legacy name, this is the squared-L2 distance
    // ||vec - query||^2.  HNSW's graph is built under L2; the layer
    // search picks the smallest distance (sort ascending) so a raw
    // dot product would select the *least* similar candidate.
    ScopedTimer timer(PerfEvent::DotProduct, vec.size());
    if (vec.size() != query.size()) {
        throw std::runtime_error("dot_product size mismatch");
    }
    if (vec.empty()) {
        return make_public_share(0, ctx.party);
    }
    std::vector<si64> diffs(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        diffs[i] = vec[i] - query[i];
    }
    auto products = mul_vec(ctx, diffs, diffs);
    return sum_vec(products, ctx.party);
}

static std::vector<si64> dot_products_for_query(MpcContext& ctx,
                                                const si64Matrix& vec_tbl,
                                                const std::vector<si64>& query) {
    u64 k = static_cast<u64>(vec_tbl.rows());
    u64 dim = static_cast<u64>(vec_tbl.cols());
    ScopedTimer timer(PerfEvent::DotProduct, k * dim);
    if (k == 0 || dim == 0) {
        return {};
    }
    if (query.size() != dim) {
        throw std::runtime_error("dot_products_for_query size mismatch");
    }

    // Compute dot products row-by-row for numerical stability
    std::vector<si64> out(k);
    for (u64 i = 0; i < k; ++i) {
        std::vector<si64> row(dim);
        for (u64 j = 0; j < dim; ++j) {
            row[j] = matrix_get(vec_tbl, i, j);
        }
        out[i] = dot_product(ctx, row, query);
    }
    return out;
}

static si64 compute_dummy_id(MpcContext& ctx, const std::vector<si64>& is_dummy) {
    si64 dummy = make_public_share(0, ctx.party);
    si64 seen = make_public_share(0, ctx.party);
    si64 one = make_public_share(1, ctx.party);
    for (size_t i = 0; i < is_dummy.size(); ++i) {
        si64 not_seen = one - seen;
        si64 take = mul_scalar(ctx, is_dummy[i], not_seen);
        dummy = dummy + mul_const(take, static_cast<i64>(i));
        si64 prod = mul_scalar(ctx, seen, take);
        seen = seen + take - prod;
    }
    return dummy;
}

static LayerData load_layer(const LayerConfig& cfg,
                            const std::string& share_root,
                            u64 party,
                            u64 dim) {
    std::string base = share_root + "/p" + std::to_string(party);
    LayerData layer;
    layer.cfg = cfg;

    layer.graph = load_share_matrix(
        base + "/layer" + std::to_string(cfg.idx) + "_neigh_s0.bin",
        base + "/layer" + std::to_string(cfg.idx) + "_neigh_s1.bin",
        cfg.n, cfg.m, cfg.id_bits);

    layer.vecs = load_share_matrix(
        base + "/layer" + std::to_string(cfg.idx) + "_vecs_s0.bin",
        base + "/layer" + std::to_string(cfg.idx) + "_vecs_s1.bin",
        cfg.n, dim, cfg.vec_bits);

    layer.is_dummy = load_share_vector(
        base + "/layer" + std::to_string(cfg.idx) + "_is_dummy_s0.bin",
        base + "/layer" + std::to_string(cfg.idx) + "_is_dummy_s1.bin",
        cfg.n, 1);

    if (cfg.down_bits > 0) {
        layer.down = load_share_vector(
            base + "/layer" + std::to_string(cfg.idx) + "_down_s0.bin",
            base + "/layer" + std::to_string(cfg.idx) + "_down_s1.bin",
            cfg.n, cfg.down_bits);
    }

    return layer;
}

static std::pair<si64, si64> layer_max_dist(MpcContext& ctx, u64 dist_bits) {
    if (dist_bits == 0 || dist_bits > 63) {
        throw std::runtime_error("inner product bitlen too large for 64-bit ring");
    }
    u64 max_key = (1ULL << dist_bits) - 1ULL;
    si64 max_dist = make_public_share(static_cast<i64>(max_key), ctx.party);
    si64 md_key = make_public_share(static_cast<i64>(max_key - 1ULL), ctx.party);
    return {max_dist, md_key};
}

static si64 search_layer_single(MpcContext& ctx,
                                ComparatorCache& cmp,
                                const LayerData& layer,
                                const si64& entry_id,
                                const std::vector<si64>& query,
                                u64 dim) {
    u64 dist_bits = dist_bits_for_layer(layer.cfg.vec_bits, dim);
    auto max_pair = layer_max_dist(ctx, dist_bits);
    si64 max_dist = max_pair.first;
    si64 md_key = max_pair.second;
    u64 visited_len = 1 + layer.cfg.t * layer.cfg.m;
    std::vector<si64> visited_ids(visited_len, make_public_share(0, ctx.party));
    std::vector<si64> visited_valid(visited_len, make_public_share(0, ctx.party));
    visited_ids[0] = entry_id;
    visited_valid[0] = make_public_share(1, ctx.party);

    std::vector<si64> c_dist(layer.cfg.l_c, max_dist);
    std::vector<si64> c_id(layer.cfg.l_c, layer.dummy_id);
    std::vector<si64> w_dist(layer.cfg.l_w, max_dist);
    std::vector<si64> w_id(layer.cfg.l_w, layer.dummy_id);

    u64 entry_open = reveal_id(ctx, entry_id, layer.cfg.id_bits);
    u64 entry_fetch = normalize_id(entry_open, layer);
    auto entry_vec = fetch_row_public(layer.vecs, entry_fetch, layer.dummy_open);
    auto entry_dist = dot_product(ctx, entry_vec, query);
    c_dist[0] = entry_dist;
    c_id[0] = entry_id;
    w_dist[0] = entry_dist;
    w_id[0] = entry_id;

    for (u64 t = 0; t < layer.cfg.t; ++t) {
        sort_pairs(ctx, cmp, c_dist, c_id, dist_bits, max_dist, layer.dummy_id);
        si64 c0_id = c_id[0];
        u64 c0_open = reveal_id(ctx, c0_id, layer.cfg.id_bits);
        u64 c0_fetch = normalize_id(c0_open, layer);
        auto neigh = fetch_row_public(layer.graph, c0_fetch, layer.dummy_open);

        auto hits = membership_batch(ctx, cmp, neigh, visited_ids, visited_valid, layer.cfg.id_bits);

        std::vector<si64> dummies(layer.cfg.m, layer.dummy_id);
        auto id_primes = select_vec_by_bit(ctx, neigh, hits, layer.dummy_id);

        auto id_primes_open = reveal_ids(ctx, id_primes, layer.cfg.id_bits);

        si64Matrix vec_tbl(layer.cfg.m, dim);
        for (u64 j = 0; j < layer.cfg.m; ++j) {
            u64 fetch_idx = normalize_id(id_primes_open[j], layer);
            auto vec_row = fetch_row_public(layer.vecs, fetch_idx, layer.dummy_open);
            for (u64 d = 0; d < dim; ++d) {
                matrix_set(vec_tbl, j, d, vec_row[d]);
            }
        }
        auto dists = dot_products_for_query(ctx, vec_tbl, query);

        std::vector<si64> md_keys(layer.cfg.m, md_key);
        auto cand_dist = select_vec_by_bit(ctx, dists, hits, md_key);
        auto cand_id = id_primes;

        for (u64 j = 0; j < layer.cfg.m; ++j) {
            size_t idx = 1 + t * layer.cfg.m + j;
            visited_ids[idx] = neigh[j];
            si64 one = make_public_share(1, ctx.party);
            visited_valid[idx] = one - hits[j];
        }

        merge_topk(ctx, cmp, c_dist, c_id, cand_dist, cand_id, layer.cfg.l_c,
                   dist_bits, max_dist, layer.dummy_id);
        merge_topk(ctx, cmp, w_dist, w_id, cand_dist, cand_id, layer.cfg.l_w,
                   dist_bits, max_dist, layer.dummy_id);
    }

    sort_pairs(ctx, cmp, w_dist, w_id, dist_bits, max_dist, layer.dummy_id);
    return w_id[0];
}

static std::vector<si64> search_layer_multi(MpcContext& ctx,
                                            ComparatorCache& cmp,
                                            const LayerData& layer,
                                            const std::vector<si64>& entry_ids,
                                            const si64Matrix& queries,
                                            u64 dim) {
    u64 dist_bits = dist_bits_for_layer(layer.cfg.vec_bits, dim);
    auto max_pair = layer_max_dist(ctx, dist_bits);
    si64 max_dist = max_pair.first;
    si64 md_key = max_pair.second;
    u64 cap_g = entry_ids.size() * layer.cfg.t;
    u64 cap_v = entry_ids.size() * layer.cfg.t * layer.cfg.m;
    u64 visited_len = 1 + layer.cfg.t * layer.cfg.m;

    std::vector<u64> vg_keys(cap_g, 0);
    si64Matrix vg_vals(cap_g, layer.cfg.m);
    std::vector<u64> vd_keys(cap_v, 0);
    si64Matrix vd_vals(cap_v, dim);

    std::vector<si64> out_entries(entry_ids.size());

    for (u64 q = 0; q < entry_ids.size(); ++q) {
        std::vector<si64> visited_ids(visited_len, make_public_share(0, ctx.party));
        std::vector<si64> visited_valid(visited_len, make_public_share(0, ctx.party));
        visited_ids[0] = entry_ids[q];
        visited_valid[0] = make_public_share(1, ctx.party);

        std::vector<si64> c_dist(layer.cfg.l_c, max_dist);
        std::vector<si64> c_id(layer.cfg.l_c, layer.dummy_id);
        std::vector<si64> w_dist(layer.cfg.l_w, max_dist);
        std::vector<si64> w_id(layer.cfg.l_w, layer.dummy_id);

        std::vector<si64> query(dim);
        for (u64 j = 0; j < dim; ++j) {
            query[j] = queries(q, j);
        }

        u64 entry_open = reveal_id(ctx, entry_ids[q], layer.cfg.id_bits);
        u64 entry_fetch = normalize_id(entry_open, layer);
        auto entry_vec = fetch_row_public(layer.vecs, entry_fetch, layer.dummy_open);
        si64 entry_dist = dot_product(ctx, entry_vec, query);
        c_dist[0] = entry_dist;
        c_id[0] = entry_ids[q];
        w_dist[0] = entry_dist;
        w_id[0] = entry_ids[q];

        for (u64 t = 0; t < layer.cfg.t; ++t) {
            sort_pairs(ctx, cmp, c_dist, c_id,
                       dist_bits, max_dist, layer.dummy_id);
            si64 c0_id = c_id[0];

            size_t hist_len_g = q * layer.cfg.t;
            u64 c0_open = reveal_id(ctx, c0_id, layer.cfg.id_bits);
            u64 c0_fetch = normalize_id(c0_open, layer);
            std::vector<si64> neigh_cache;
            bool hit_g_plain = lookup_graph_cache_plain(vg_keys, vg_vals, hist_len_g, c0_open, neigh_cache);
            std::vector<si64> neigh_tbl;
            if (!hit_g_plain) {
                neigh_tbl = fetch_row_public(layer.graph, c0_fetch, layer.dummy_open);
            }
            std::vector<si64> neigh_sel = hit_g_plain ? neigh_cache : neigh_tbl;

            std::vector<si64> cand_dist(layer.cfg.m);
            std::vector<si64> cand_id(layer.cfg.m);
            size_t hist_len_v = q * layer.cfg.t * layer.cfg.m;

            auto hits = membership_batch(ctx, cmp, neigh_sel, visited_ids, visited_valid, layer.cfg.id_bits);
            auto id_primes = select_vec_by_bit(ctx, neigh_sel, hits, layer.dummy_id);
            auto id_primes_open = reveal_ids(ctx, id_primes, layer.cfg.id_bits);

            si64Matrix vec_tbl_batch(layer.cfg.m, dim);
            std::vector<bool> cache_hits(layer.cfg.m);
            std::vector<std::vector<si64>> vec_cache_all(layer.cfg.m);
            for (u64 j = 0; j < layer.cfg.m; ++j) {
                u64 id_fetch = normalize_id(id_primes_open[j], layer);
                std::vector<si64> vec_cache;
                bool hit_v_plain = lookup_vector_cache_plain(vd_keys, vd_vals, hist_len_v,
                                                             id_primes_open[j], vec_cache);
                cache_hits[j] = hit_v_plain;
                std::vector<si64> vec_row;
                if (hit_v_plain) {
                    vec_row = vec_cache;
                    vec_cache_all[j] = vec_cache;
                } else {
                    vec_row = fetch_row_public(layer.vecs, id_fetch, layer.dummy_open);
                    vec_cache_all[j] = vec_row;
                }
                for (u64 d = 0; d < dim; ++d) {
                    matrix_set(vec_tbl_batch, j, d, vec_row[d]);
                }
            }

            auto dists = dot_products_for_query(ctx, vec_tbl_batch, query);
            auto cand_dist_sel = select_vec_by_bit(ctx, dists, hits, md_key);
            cand_dist = cand_dist_sel;
            cand_id = id_primes;

            for (u64 j = 0; j < layer.cfg.m; ++j) {
                size_t idx = 1 + t * layer.cfg.m + j;
                visited_ids[idx] = neigh_sel[j];
                visited_valid[idx] = make_public_share(1, ctx.party) - hits[j];

                size_t v_idx = q * layer.cfg.t * layer.cfg.m + t * layer.cfg.m + j;
                vd_keys[v_idx] = id_primes_open[j];
                for (u64 d = 0; d < dim; ++d) {
                    matrix_set(vd_vals, v_idx, d, vec_cache_all[j][d]);
                }
            }

            size_t g_idx = q * layer.cfg.t + t;
            vg_keys[g_idx] = c0_open;
            for (u64 j = 0; j < layer.cfg.m; ++j) {
                matrix_set(vg_vals, g_idx, j, hit_g_plain ? neigh_cache[j] : neigh_tbl[j]);
            }

            merge_topk(ctx, cmp, c_dist, c_id, cand_dist, cand_id, layer.cfg.l_c,
                       dist_bits, max_dist, layer.dummy_id);
            merge_topk(ctx, cmp, w_dist, w_id, cand_dist, cand_id, layer.cfg.l_w,
                       dist_bits, max_dist, layer.dummy_id);
        }

        sort_pairs(ctx, cmp, w_dist, w_id,
                   dist_bits, max_dist, layer.dummy_id);
        out_entries[q] = w_id[0];
    }
    return out_entries;
}

static std::vector<si64> search_layer_batch(MpcContext& ctx,
                                            ComparatorCache& cmp,
                                            const LayerData& layer,
                                            const std::vector<si64>& entry_ids,
                                            const si64Matrix& queries,
                                            u64 dim,
                                            bool progress) {
    u64 dist_bits = dist_bits_for_layer(layer.cfg.vec_bits, dim);
    auto max_pair = layer_max_dist(ctx, dist_bits);
    si64 max_dist = max_pair.first;
    si64 md_key = max_pair.second;
    u64 k = entry_ids.size() * layer.cfg.m;
    std::vector<u64> visited_open;
    visited_open.reserve(entry_ids.size() + layer.cfg.t * k);
    std::unordered_set<u64> visited_set;
    std::vector<u64> visited_empty;

    auto entry_open = reveal_ids(ctx, entry_ids, layer.cfg.id_bits);
    if (progress) {
        std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " entry_open=";
        for (size_t i = 0; i < entry_open.size(); ++i) {
            std::cout << entry_open[i] << (i + 1 == entry_open.size() ? "" : ",");
        }
        std::cout << " n_real=" << layer.cfg.n_real << std::endl;
    }
    for (u64 q = 0; q < entry_ids.size(); ++q) {
        u64 entry_id_open = entry_open[q];
        if (entry_id_open < layer.cfg.n_real && visited_set.insert(entry_id_open).second) {
            visited_open.push_back(entry_id_open);
        }
    }

    si64Matrix c_dist(entry_ids.size(), layer.cfg.l_c);
    si64Matrix c_id(entry_ids.size(), layer.cfg.l_c);
    si64Matrix w_dist(entry_ids.size(), layer.cfg.l_w);
    si64Matrix w_id(entry_ids.size(), layer.cfg.l_w);

    for (u64 q = 0; q < entry_ids.size(); ++q) {
        for (u64 i = 0; i < layer.cfg.l_c; ++i) {
            matrix_set(c_dist, q, i, max_dist);
            matrix_set(c_id, q, i, layer.dummy_id);
        }
        for (u64 i = 0; i < layer.cfg.l_w; ++i) {
            matrix_set(w_dist, q, i, max_dist);
            matrix_set(w_id, q, i, layer.dummy_id);
        }
        std::vector<si64> query(dim);
        for (u64 j = 0; j < dim; ++j) {
            query[j] = matrix_get(queries, q, j);
        }
        u64 entry_fetch = normalize_id(entry_open[q], layer);
        auto entry_vec = fetch_row_public(layer.vecs, entry_fetch, layer.dummy_open);
        si64 entry_dist = dot_product(ctx, entry_vec, query);
        matrix_set(c_dist, q, 0, entry_dist);
        matrix_set(c_id, q, 0, entry_ids[q]);
        matrix_set(w_dist, q, 0, entry_dist);
        matrix_set(w_id, q, 0, entry_ids[q]);
    }

    std::vector<si64> u(entry_ids.begin(), entry_ids.end());

    for (u64 t = 0; t < layer.cfg.t; ++t) {
        if (progress && aby3_debug_enabled()) {
            std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch t=" << t
                      << " visited=" << visited_open.size() << " k=" << k << std::endl;
        }
        std::vector<si64> cand_c0(u.begin(), u.end());
        std::vector<si64> dummy_c0(entry_ids.size());
        u64 dummy_base_c0 = layer.cfg.n_real + t * entry_ids.size();
        u64 id_mask = mask_for_bits(layer.cfg.id_bits);
        for (u64 q = 0; q < entry_ids.size(); ++q) {
            u64 dummy_val = dummy_base_c0 + q;
            if (dummy_val > id_mask) {
                dummy_val = id_mask;
            }
            dummy_c0[q] = make_public_share(static_cast<i64>(dummy_val), ctx.party);
        }
        const std::vector<u64>& visited_c0 = (t == 0) ? visited_empty : visited_open;
        BatchDedupResult dedup_c0 = batch_dedup(ctx, cmp, cand_c0, dummy_c0, visited_c0,
                                                layer.cfg.id_bits);

        auto dedup_c0_open = reveal_ids(ctx, dedup_c0.ids, layer.cfg.id_bits);
        if (progress && aby3_debug_enabled()) {
            std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch t=" << t
                      << " dedup_c0_open=";
            for (size_t i = 0; i < dedup_c0_open.size(); ++i) {
                std::cout << dedup_c0_open[i] << (i + 1 == dedup_c0_open.size() ? "" : ",");
            }
            std::cout << std::endl;
        }
        std::vector<si64> neigh_flat(k);
        for (u64 q = 0; q < entry_ids.size(); ++q) {
            u64 c0_fetch = normalize_id(dedup_c0_open[q], layer);
            auto neigh = fetch_row_public(layer.graph, c0_fetch, layer.dummy_open);
            std::vector<si64> hits(layer.cfg.m, dedup_c0.hits[q]);
            auto neigh_sel = select_vec_by_bit(ctx, neigh, hits, layer.dummy_id);
            for (u64 j = 0; j < layer.cfg.m; ++j) {
                neigh_flat[q * layer.cfg.m + j] = neigh_sel[j];
            }
        }

        std::vector<si64> dummy_neigh(k);
        u64 dummy_base_neigh = layer.cfg.n_real + layer.cfg.t * entry_ids.size() + t * k;
        for (u64 i = 0; i < k; ++i) {
            u64 dummy_val = dummy_base_neigh + i;
            if (dummy_val > id_mask) {
                dummy_val = id_mask;
            }
            dummy_neigh[i] = make_public_share(static_cast<i64>(dummy_val), ctx.party);
        }
        BatchDedupResult dedup_neigh = batch_dedup(ctx, cmp, neigh_flat, dummy_neigh,
                                                   visited_open, layer.cfg.id_bits);

        auto id_fetch_vec = select_vec_by_bit(ctx, dedup_neigh.ids, dedup_neigh.hits, layer.dummy_id);
        auto fetched_open = reveal_ids(ctx, id_fetch_vec, layer.cfg.id_bits);
        if (progress && aby3_debug_enabled()) {
            std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch t=" << t
                      << " fetched_open=";
            for (size_t i = 0; i < fetched_open.size(); ++i) {
                std::cout << fetched_open[i] << (i + 1 == fetched_open.size() ? "" : ",");
            }
            std::cout << std::endl;
        }
        si64Matrix vec_tbl(k, dim);
        for (u64 i = 0; i < k; ++i) {
            u64 id_fetch_norm = normalize_id(fetched_open[i], layer);
            auto row = fetch_row_public(layer.vecs, id_fetch_norm, layer.dummy_open);
            for (u64 j = 0; j < dim; ++j) {
                matrix_set(vec_tbl, i, j, row[j]);
            }
        }
        for (u64 i = 0; i < k; ++i) {
            u64 id_open = fetched_open[i];
            if (id_open < layer.cfg.n_real && visited_set.insert(id_open).second) {
                visited_open.push_back(id_open);
            }
        }

        for (u64 q = 0; q < entry_ids.size(); ++q) {
            std::vector<si64> cand_id(k);
            std::vector<si64> query(dim);
            for (u64 j = 0; j < dim; ++j) {
                query[j] = queries(q, j);
            }
            auto dist_vec = dot_products_for_query(ctx, vec_tbl, query);
            auto cand_dist = select_vec_by_bit(ctx, dist_vec, dedup_neigh.hits, md_key);
            for (u64 i = 0; i < k; ++i) {
                cand_id[i] = dedup_neigh.ids[i];
            }

            std::vector<si64> cur_dist(layer.cfg.l_c);
            std::vector<si64> cur_id(layer.cfg.l_c);
            std::vector<si64> cur_w_dist(layer.cfg.l_w);
            std::vector<si64> cur_w_id(layer.cfg.l_w);
            for (u64 i = 0; i < layer.cfg.l_c; ++i) {
                cur_dist[i] = matrix_get(c_dist, q, i);
                cur_id[i] = matrix_get(c_id, q, i);
            }
            for (u64 i = 0; i < layer.cfg.l_w; ++i) {
                cur_w_dist[i] = matrix_get(w_dist, q, i);
                cur_w_id[i] = matrix_get(w_id, q, i);
            }
            if (progress && aby3_debug_enabled() && t == 0) {
                auto cand_id_open = reveal_ids(ctx, cand_id, layer.cfg.id_bits);
                auto cur_id_open = reveal_ids(ctx, cur_id, layer.cfg.id_bits);
                std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch t=" << t
                          << " q=" << q << " cand_id=";
                for (size_t i = 0; i < cand_id_open.size(); ++i) {
                    std::cout << cand_id_open[i] << (i + 1 == cand_id_open.size() ? "" : ",");
                }
                std::cout << " cur_id=";
                for (size_t i = 0; i < cur_id_open.size(); ++i) {
                    std::cout << cur_id_open[i] << (i + 1 == cur_id_open.size() ? "" : ",");
                }
                std::cout << std::endl;
            }

            merge_topk(ctx, cmp, cur_dist, cur_id, cand_dist, cand_id, layer.cfg.l_c,
                       dist_bits, max_dist, layer.dummy_id);
            merge_topk(ctx, cmp, cur_w_dist, cur_w_id, cand_dist, cand_id, layer.cfg.l_w,
                       dist_bits, max_dist, layer.dummy_id);

            if (progress && aby3_debug_enabled() && t == 0) {
                auto cur_id_open = reveal_ids(ctx, cur_id, layer.cfg.id_bits);
                auto cur_w_id_open = reveal_ids(ctx, cur_w_id, layer.cfg.id_bits);
                std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch t=" << t
                          << " q=" << q << " cur_id_post=";
                for (size_t i = 0; i < cur_id_open.size(); ++i) {
                    std::cout << cur_id_open[i] << (i + 1 == cur_id_open.size() ? "" : ",");
                }
                std::cout << " cur_w_id_post=";
                for (size_t i = 0; i < cur_w_id_open.size(); ++i) {
                    std::cout << cur_w_id_open[i] << (i + 1 == cur_w_id_open.size() ? "" : ",");
                }
                std::cout << std::endl;
            }

            for (u64 i = 0; i < layer.cfg.l_c; ++i) {
                matrix_set(c_dist, q, i, cur_dist[i]);
                matrix_set(c_id, q, i, cur_id[i]);
            }
            for (u64 i = 0; i < layer.cfg.l_w; ++i) {
                matrix_set(w_dist, q, i, cur_w_dist[i]);
                matrix_set(w_id, q, i, cur_w_id[i]);
            }
            u[q] = c_id(q, 0);
        }
        if (progress && aby3_debug_enabled()) {
            auto u_open = reveal_ids(ctx, u, layer.cfg.id_bits);
            std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch t=" << t
                      << " u_open=";
            for (size_t i = 0; i < u_open.size(); ++i) {
                std::cout << u_open[i] << (i + 1 == u_open.size() ? "" : ",");
            }
            std::cout << std::endl;
        }
    }

    std::vector<si64> out_entries(entry_ids.size());
    for (u64 q = 0; q < entry_ids.size(); ++q) {
        std::vector<si64> cur_dist(layer.cfg.l_w);
        std::vector<si64> cur_id(layer.cfg.l_w);
        for (u64 i = 0; i < layer.cfg.l_w; ++i) {
            cur_dist[i] = w_dist(q, i);
            cur_id[i] = w_id(q, i);
        }
        sort_pairs(ctx, cmp, cur_dist, cur_id,
                   dist_bits, max_dist, layer.dummy_id);
        out_entries[q] = cur_id[0];
    }
    if (progress && aby3_debug_enabled()) {
        auto out_open = reveal_ids(ctx, out_entries, layer.cfg.id_bits);
        std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx << " batch out=";
        for (size_t i = 0; i < out_open.size(); ++i) {
            std::cout << out_open[i] << (i + 1 == out_open.size() ? "" : ",");
        }
        std::cout << std::endl;
    }
    return out_entries;
}

static void setup_context(MpcContext& ctx, const std::string& host, u64 port) {
    CommPkg comm;
    if (ctx.party == 0) {
        comm.mNext = Session(ctx.ios, host + ":" + std::to_string(port), SessionMode::Server, "01").addChannel();
        comm.mPrev = Session(ctx.ios, host + ":" + std::to_string(port + 1), SessionMode::Server, "02").addChannel();
    } else if (ctx.party == 1) {
        comm.mNext = Session(ctx.ios, host + ":" + std::to_string(port + 2), SessionMode::Server, "12").addChannel();
        comm.mPrev = Session(ctx.ios, host + ":" + std::to_string(port), SessionMode::Client, "01").addChannel();
    } else {
        comm.mNext = Session(ctx.ios, host + ":" + std::to_string(port + 1), SessionMode::Client, "02").addChannel();
        comm.mPrev = Session(ctx.ios, host + ":" + std::to_string(port + 2), SessionMode::Client, "12").addChannel();
    }

    ctx.enc.init(ctx.party, comm, sysRandomSeed());
    ctx.eval.init(ctx.party, comm, sysRandomSeed());
    ctx.rt.init(ctx.party, comm);
    ctx.conv.init(ctx.rt, ctx.enc.mShareGen);
}

static void write_output_shares(const std::string& out_dir,
                                const std::string& name,
                                const std::vector<si64>& values,
                                u64 bitlen) {
    std::vector<i64> s0(values.size());
    std::vector<i64> s1(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        s0[i] = values[i][0];
        s1[i] = values[i][1];
    }
    write_bin(out_dir + "/" + name + "_s0.bin", s0, bitlen);
    write_bin(out_dir + "/" + name + "_s1.bin", s1, bitlen);
}

int main(int argc, char** argv) {
    try {
        CLP cmd(argc, argv);
        u64 party = cmd.getOr<u64>("party", 0);
        std::string cfg_path = cmd.getOr<std::string>("cfg", "");
        std::string share_root = cmd.getOr<std::string>("shares", "");
        std::string out_root = cmd.getOr<std::string>("out", "");
        std::string mode = cmd.getOr<std::string>("mode", "single");
        u64 num_queries = cmd.getOr<u64>("num_queries", 0);
        u64 entry_count = cmd.getOr<u64>("entry_count", 0);
        std::string output_mode = cmd.getOr<std::string>("output", "id");
        std::string host = cmd.getOr<std::string>("host", "127.0.0.1");
        u64 port = cmd.getOr<u64>("port", 1313);
        bool progress = cmd.isSet("progress");
        bool profile = cmd.isSet("profile");

        if (cfg_path.empty() || share_root.empty() || out_root.empty()) {
            std::cerr << "missing required args: -cfg, -shares, -out" << std::endl;
            return 1;
        }
        if (party > 2) {
            std::cerr << "party must be 0/1/2" << std::endl;
            return 1;
        }
        if (mode != "single" && mode != "multi" && mode != "batch") {
            std::cerr << "mode must be single|multi|batch" << std::endl;
            return 1;
        }
        bool output_id = (output_mode == "id" || output_mode == "ids" || output_mode == "both");
        bool output_vec = (output_mode == "vec" || output_mode == "vector" || output_mode == "both");
        if (!output_id && !output_vec) {
            std::cerr << "output must be id|vec|both" << std::endl;
            return 1;
        }

        std::ifstream cfg_file(cfg_path);
        if (!cfg_file.is_open()) {
            std::cerr << "failed to open cfg: " << cfg_path << std::endl;
            return 1;
        }
        json cfg_json = json::parse(cfg_file);

        u64 dim = cfg_json.at("D").get<u64>();
        if (dim == 0) {
            std::cerr << "D must be > 0 in cfg" << std::endl;
            return 1;
        }
        if (num_queries == 0) {
            num_queries = cfg_json.value("num_queries", 1);
        }
        if (num_queries == 0) {
            std::cerr << "num_queries must be > 0" << std::endl;
            return 1;
        }
        if (mode == "single" && num_queries != 1) {
            std::cerr << "single mode requires num_queries=1" << std::endl;
            return 1;
        }
        if (entry_count == 0) {
            entry_count = (mode == "single") ? 1 : num_queries;
        }
        if (entry_count != 1 && entry_count != num_queries) {
            std::cerr << "entry_count must be 1 or num_queries" << std::endl;
            return 1;
        }

        std::string meta_path = share_root + "/p0/meta.json";
        std::ifstream meta_file(meta_path);
        if (!meta_file.is_open()) {
            std::cerr << "failed to open share meta: " << meta_path << std::endl;
            return 1;
        }
        json meta_json = json::parse(meta_file);
        if (meta_json.value("share_format", "") != "aby3") {
            std::cerr << "share_format must be aby3 in " << meta_path << std::endl;
            return 1;
        }

        std::unordered_map<u64, LayerConfig> share_cfg;
        for (const auto& layer : meta_json.at("layers")) {
            LayerConfig cfg;
            cfg.idx = layer.at("idx").get<u64>();
            cfg.n = layer.at("N").get<u64>();
            cfg.n_real = layer.value("N_real", cfg.n);
            cfg.m = layer.at("M").get<u64>();
            cfg.id_bits = layer.at("id_bits").get<u64>();
            cfg.vec_bits = layer.at("vec_bits").get<u64>();
            cfg.x2_bits = layer.at("x2_bits").get<u64>();
            cfg.down_bits = layer.at("down_bits").get<u64>();
            share_cfg[cfg.idx] = cfg;
        }

        std::vector<LayerData> layers;
        for (const auto& layer : cfg_json.at("layers")) {
            LayerConfig cfg;
            cfg.idx = layer.at("idx").get<u64>();
            auto it = share_cfg.find(cfg.idx);
            if (it == share_cfg.end()) {
                std::cerr << "missing share metadata for layer idx=" << cfg.idx << std::endl;
                return 1;
            }
            const LayerConfig& share_layer = it->second;
            cfg.n = layer.at("N").get<u64>();
            cfg.m = layer.at("M").get<u64>();
            cfg.t = layer.at("T").get<u64>();
            cfg.l_c = layer.at("L_C").get<u64>();
            cfg.l_w = layer.at("L_W").get<u64>();
            cfg.id_bits = layer.at("id_bits").get<u64>();
            cfg.vec_bits = layer.value("vec_bits", share_layer.vec_bits);
            cfg.x2_bits = layer.value("x2_bits", share_layer.x2_bits);
            cfg.down_bits = share_layer.down_bits;
            cfg.n_real = layer.value("N_real", share_layer.n_real ? share_layer.n_real : cfg.n);
            if (cfg.n_real == 0 || cfg.n_real > cfg.n) {
                cfg.n_real = cfg.n;
            }

            LayerData layer_data = load_layer(cfg, share_root, party, dim);
            layers.push_back(std::move(layer_data));
        }

        MpcContext ctx;
        ctx.party = party;
        setup_context(ctx, host, port);
        ComparatorCache cmp(ctx);
        g_perf.enabled = profile;

        if (layers.empty()) {
            std::cerr << "config has no layers" << std::endl;
            return 1;
        }

        for (auto& layer : layers) {
            layer.dummy_id = compute_dummy_id(ctx, layer.is_dummy);
            layer.dummy_open = reveal_id(ctx, layer.dummy_id, layer.cfg.id_bits);
            if (layer.cfg.n > 0 && layer.dummy_open >= layer.cfg.n) {
                layer.dummy_open = layer.cfg.n - 1;
            }
            if (progress && aby3_debug_enabled()) {
                std::cout << "[aby3 p" << ctx.party << "] layer=" << layer.cfg.idx
                          << " dummy_open=" << layer.dummy_open << std::endl;
            }
        }

        u64 vec_bits = layers.front().cfg.vec_bits;
        size_t query_count = num_queries * dim;
        std::string base = share_root + "/p" + std::to_string(party);
        auto query_flat = load_share_vector(base + "/queries_s0.bin",
                                            base + "/queries_s1.bin",
                                            query_count,
                                            vec_bits);
        si64Matrix queries = vector_to_matrix(query_flat, num_queries, dim);

        auto entry_bits = layers.front().cfg.id_bits;
        std::string entry_s0 = base + "/entry_point_top_local_s0.bin";
        std::string entry_s1 = base + "/entry_point_top_local_s1.bin";
        size_t entry_file_count = file_entry_count(entry_s0, entry_bits);
        size_t entry_file_count_1 = file_entry_count(entry_s1, entry_bits);
        if (entry_file_count != entry_file_count_1) {
            std::cerr << "entry share size mismatch between s0 and s1" << std::endl;
            return 1;
        }

        std::vector<si64> entry_vec;
        if (entry_file_count == 1 && entry_count == num_queries) {
            entry_vec = load_share_vector(entry_s0, entry_s1, 1, entry_bits);
            entry_vec.resize(num_queries, entry_vec[0]);
        } else {
            if (entry_file_count != entry_count) {
                std::cerr << "entry share size " << entry_file_count
                          << " does not match entry_count=" << entry_count << std::endl;
                return 1;
            }
            entry_vec = load_share_vector(entry_s0, entry_s1, entry_count, entry_bits);
            if (entry_count == 1 && num_queries > 1) {
                entry_vec.resize(num_queries, entry_vec[0]);
            }
        }

        if (progress && party == 0) {
            std::cout << "[aby3] start mode=" << mode
                      << " num_queries=" << num_queries
                      << " layers=" << layers.size() << std::endl;
        }

        std::vector<si64> current_entries = entry_vec;

        auto online_start = std::chrono::steady_clock::now();
        CommStats comm_start = get_comm_stats(ctx);

        for (size_t li = 0; li < layers.size(); ++li) {
            if (progress && party == 0) {
                auto& cfg = layers[li].cfg;
                std::cout << "[aby3] layer=" << li << " start n=" << cfg.n
                          << " m=" << cfg.m << " t=" << cfg.t
                          << " l_c=" << cfg.l_c << " l_w=" << cfg.l_w << std::endl;
            }
            if (profile) {
                perf_reset();
            }

            if (mode == "single") {
                std::vector<si64> query(dim);
                for (u64 j = 0; j < dim; ++j) {
                    query[j] = queries(0, j);
                }
                si64 best = search_layer_single(ctx, cmp, layers[li], current_entries[0],
                                                query, dim);
                current_entries.assign(1, best);
            } else if (mode == "multi") {
                current_entries = search_layer_multi(ctx, cmp, layers[li], current_entries, queries, dim);
            } else {
                current_entries = search_layer_batch(ctx, cmp, layers[li], current_entries, queries, dim, progress);
            }

            if (li + 1 < layers.size()) {
                std::vector<si64> next_entries(num_queries);
                for (u64 q = 0; q < num_queries; ++q) {
                    u64 entry_open = reveal_id(ctx, current_entries[q], layers[li].cfg.id_bits);
                    u64 entry_fetch = normalize_id(entry_open, layers[li]);
                    next_entries[q] = fetch_value_public(layers[li].down, entry_fetch, layers[li].dummy_open);
                }
                current_entries = next_entries;
            }

            if (progress && party == 0) {
                std::cout << "[aby3] layer=" << li << " done" << std::endl;
            }
            if (profile) {
                perf_print("layer=" + std::to_string(li), ctx.party);
            }
        }

        auto online_end = std::chrono::steady_clock::now();
        CommStats comm_end = get_comm_stats(ctx);
        double online_sec = std::chrono::duration<double>(online_end - online_start).count();
        // Per-party online bytes = sent + received across both channels.
        // Rationale: matches 2PC convention (hnsecw_single_b2y.cpp:25).
        // Addresses reviewer R3.A1 (fix: was bytes_sent only).
        u64 online_bytes = comm_end.total() - comm_start.total();
        u64 online_sent = comm_end.sent() - comm_start.sent();
        u64 online_recv = comm_end.recv() - comm_start.recv();

        // All three parties print their own local (sent, recv, sent+recv).
        // Post-processing can then compute aggregate_wire_mb = sum(local_total_i) / 2
        // and max_party_local_mb = max_i(local_total_i).
        // Rationale: adversarial review flagged that reporting only party 0 left the
        // symmetry claim unauditable in logs
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[Party " << party << " Online] sent_MB="
                  << (online_sent / (1024.0 * 1024.0))
                  << " recv_MB=" << (online_recv / (1024.0 * 1024.0))
                  << " total_MB=" << (online_bytes / (1024.0 * 1024.0))
                  << " latency_s=" << online_sec
                  << std::endl;

        if (party == 0) {
            // Backward-compatible summary line expected by parsers.
            std::cout << "[Total Online] latency(s)=" << online_sec
                      << " comm(MB)=" << (online_bytes / (1024.0 * 1024.0))
                      << " (party-0 local; see [Party i Online] lines for aggregation)"
                      << std::endl;
        }

        std::string out_dir = out_root + "/p" + std::to_string(party);
        if (output_id) {
            write_output_shares(out_dir, "out_ids", current_entries, layers.back().cfg.id_bits);
        }
        if (output_vec) {
            std::vector<si64> vec_out(num_queries * dim);
            for (u64 q = 0; q < num_queries; ++q) {
                u64 out_open = reveal_id(ctx, current_entries[q], layers.back().cfg.id_bits);
                u64 out_fetch = normalize_id(out_open, layers.back());
                auto vec = fetch_row_public(layers.back().vecs, out_fetch, layers.back().dummy_open);
                for (u64 j = 0; j < dim; ++j) {
                    vec_out[q * dim + j] = vec[j];
                }
            }
            write_output_shares(out_dir, "out_vecs", vec_out, layers.back().cfg.vec_bits);
        }

        if (progress && party == 0) {
            std::cout << "[aby3] done" << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
