#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include "../../abycore/aby/abyparty.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/sharing/sharing.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct PayloadArray {
    std::vector<uint32_t> wires;
    uint32_t bitlen = 0;
};

static bool reshuffle_debug_enabled() {
    return std::getenv("HNSECW_RESHUFFLE_DEBUG") != nullptr;
}

static void reshuffle_debug(const char* msg) {
    if (reshuffle_debug_enabled()) {
        std::cerr << msg << std::endl;
    }
}

static uint32_t floor_log2(uint32_t x) {
    uint32_t r = 0;
    while (x >>= 1) {
        r++;
    }
    return r;
}

static uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) {
        return 1;
    }
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

static uint32_t byte_width(uint32_t bitlen) {
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
static std::vector<uint64_t> read_bin_t(const std::string& path, size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open input: " + path);
    }
    std::vector<T> buf(count);
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) {
        throw std::runtime_error("failed to read input: " + path);
    }
    std::vector<uint64_t> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint64_t>(buf[i]);
    }
    return out;
}

static std::vector<uint64_t> read_bin(const std::string& path, size_t count, uint32_t bitlen) {
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

template <typename T>
static void write_bin_t(const std::string& path, const std::vector<uint64_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output: " + path);
    }
    std::vector<T> buf(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        buf[i] = static_cast<T>(data[i]);
    }
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(T)));
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

static void write_bin(const std::string& path, const std::vector<uint64_t>& data, uint32_t bitlen) {
    switch (byte_width(bitlen)) {
    case 1:
        write_bin_t<uint8_t>(path, data);
        break;
    case 2:
        write_bin_t<uint16_t>(path, data);
        break;
    case 4:
        write_bin_t<uint32_t>(path, data);
        break;
    default:
        write_bin_t<uint64_t>(path, data);
        break;
    }
}

static std::vector<uint64_t> apply_perm_rows(
    const std::vector<uint64_t>& data,
    const std::vector<uint64_t>& perm,
    size_t stride) {

    if (stride == 0) {
        return data;
    }
    size_t n = perm.size();
    std::vector<uint64_t> out(data.size());
    for (size_t i = 0; i < n; ++i) {
        size_t src = static_cast<size_t>(perm[i]);
        size_t dst_off = i * stride;
        size_t src_off = src * stride;
        for (size_t j = 0; j < stride; ++j) {
            out[dst_off + j] = data[src_off + j];
        }
    }
    return out;
}

static std::vector<uint64_t> apply_perm_1d(
    const std::vector<uint64_t>& data,
    const std::vector<uint64_t>& perm) {

    size_t n = perm.size();
    std::vector<uint64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t src = static_cast<size_t>(perm[i]);
        out[i] = data[src];
    }
    return out;
}

static uint32_t pack_yao_value(share* s, uint32_t bitlen, BooleanCircuit* circ) {
    std::vector<uint32_t> bits(bitlen);
    for (uint32_t i = 0; i < bitlen; ++i) {
        bits[i] = s->get_wire_id(i);
    }
    return circ->PutCombinerGate(bits);
}

static std::vector<uint32_t> cond_swap(uint32_t a, uint32_t b, uint32_t s, BooleanCircuit* circ) {
    std::vector<uint32_t> avec(1, a);
    std::vector<uint32_t> bvec(1, b);
    std::vector<std::vector<uint32_t>> out = circ->PutCondSwapGate(avec, bvec, s, true);
    return {out[0][0], out[1][0]};
}

static void bitonic_merge_multi(
    std::vector<uint32_t>& keys,
    std::vector<PayloadArray>& payloads,
    uint32_t key_bitlen,
    BooleanCircuit* circ) {

    uint32_t n = keys.size();
    if (n <= 1) {
        return;
    }
    for (const auto& p : payloads) {
        assert(p.wires.size() == n);
    }

    std::vector<uint32_t> c_keys = keys;
    std::vector<PayloadArray> c_payloads = payloads;

    std::vector<uint32_t> compa(n / 2);
    std::vector<uint32_t> compb(n / 2);
    std::vector<uint32_t> parenta(n / 2);
    std::vector<uint32_t> parentb(n / 2);
    std::vector<uint32_t> tempcmpveca(key_bitlen);
    std::vector<uint32_t> tempcmpvecb(key_bitlen);

    for (uint32_t i = 1U << floor_log2(n - 1); i > 0; i >>= 1) {
        uint32_t ctr = 0;
        for (int32_t j = (int32_t)n - 1; j >= 0; j -= (int32_t)(2 * i)) {
            for (uint32_t k = 0; k < i && j - (int32_t)i - (int32_t)k >= 0; k++) {
                compa[ctr] = (uint32_t)(j - (int32_t)i - (int32_t)k);
                compb[ctr] = (uint32_t)(j - (int32_t)k);
                ctr++;
            }
        }

        for (uint32_t l = 0; l < key_bitlen; l++) {
            for (uint32_t k = 0; k < ctr; k++) {
                parenta[k] = c_keys[compa[k]];
                parentb[k] = c_keys[compb[k]];
            }
            tempcmpveca[l] = circ->PutCombineAtPosGate(parenta, l);
            tempcmpvecb[l] = circ->PutCombineAtPosGate(parentb, l);
        }

        uint32_t selbitsvec = circ->PutGTGate(tempcmpveca, tempcmpvecb);
        std::vector<uint32_t> selbits = circ->PutSplitterGate(selbitsvec);

        for (uint32_t k = 0; k < ctr; k++) {
            auto swapped = cond_swap(c_keys[compa[k]], c_keys[compb[k]], selbits[k], circ);
            c_keys[compa[k]] = swapped[0];
            c_keys[compb[k]] = swapped[1];

            for (auto& payload : c_payloads) {
                swapped = cond_swap(payload.wires[compa[k]], payload.wires[compb[k]], selbits[k], circ);
                payload.wires[compa[k]] = swapped[0];
                payload.wires[compb[k]] = swapped[1];
            }
        }
    }

    keys.swap(c_keys);
    payloads.swap(c_payloads);
}

static void batch_compare_swap_multi(
    std::vector<uint32_t>& keys,
    std::vector<PayloadArray>& payloads,
    const std::vector<uint32_t>& compa,
    const std::vector<uint32_t>& compb,
    uint32_t key_bitlen,
    BooleanCircuit* circ) {

    size_t ctr = compa.size();
    if (ctr == 0) {
        return;
    }
    assert(compb.size() == ctr);

    std::vector<uint32_t> parenta(ctr);
    std::vector<uint32_t> parentb(ctr);
    std::vector<uint32_t> tempcmpveca(key_bitlen);
    std::vector<uint32_t> tempcmpvecb(key_bitlen);

    for (uint32_t l = 0; l < key_bitlen; l++) {
        for (size_t k = 0; k < ctr; k++) {
            parenta[k] = keys[compa[k]];
            parentb[k] = keys[compb[k]];
        }
        tempcmpveca[l] = circ->PutCombineAtPosGate(parenta, l);
        tempcmpvecb[l] = circ->PutCombineAtPosGate(parentb, l);
    }

    uint32_t selbitsvec = circ->PutGTGate(tempcmpveca, tempcmpvecb);
    std::vector<uint32_t> selbits = circ->PutSplitterGate(selbitsvec);

    for (size_t k = 0; k < ctr; k++) {
        auto swapped = cond_swap(keys[compa[k]], keys[compb[k]], selbits[k], circ);
        keys[compa[k]] = swapped[0];
        keys[compb[k]] = swapped[1];

        for (auto& payload : payloads) {
            swapped = cond_swap(payload.wires[compa[k]], payload.wires[compb[k]], selbits[k], circ);
            payload.wires[compa[k]] = swapped[0];
            payload.wires[compb[k]] = swapped[1];
        }
    }
}

static void bitonic_full_sort_multi(
    std::vector<uint32_t>& keys,
    std::vector<PayloadArray>& payloads,
    uint32_t key_bitlen,
    BooleanCircuit* circ) {

    uint32_t n = keys.size();
    if (n <= 1) {
        return;
    }

    uint32_t orig_n = n;
    if ((n & (n - 1)) != 0) {
        uint32_t padded = next_power_of_two(n);
        share* max_val = circ->PutCONSGate((uint64_t)-1, key_bitlen);
        uint32_t max_wire = pack_yao_value(max_val, key_bitlen, circ);
        std::vector<uint32_t> zero_wires;
        zero_wires.reserve(payloads.size());
        for (const auto& p : payloads) {
            share* zero = circ->PutCONSGate((uint64_t)0, p.bitlen);
            zero_wires.push_back(pack_yao_value(zero, p.bitlen, circ));
        }

        while (keys.size() < padded) {
            keys.push_back(max_wire);
            for (size_t i = 0; i < payloads.size(); ++i) {
                payloads[i].wires.push_back(zero_wires[i]);
            }
        }

        n = padded;
    }

    for (uint32_t k = 2; k <= n; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            std::vector<uint32_t> compa_up;
            std::vector<uint32_t> compb_up;
            std::vector<uint32_t> compa_down;
            std::vector<uint32_t> compb_down;
            compa_up.reserve(n / 2);
            compb_up.reserve(n / 2);
            compa_down.reserve(n / 2);
            compb_down.reserve(n / 2);

            for (uint32_t i = 0; i < n; i++) {
                uint32_t ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        compa_up.push_back(i);
                        compb_up.push_back(ixj);
                    } else {
                        compa_down.push_back(i);
                        compb_down.push_back(ixj);
                    }
                }
            }

            batch_compare_swap_multi(keys, payloads, compa_up, compb_up, key_bitlen, circ);
            if (!compa_down.empty()) {
                batch_compare_swap_multi(keys, payloads, compb_down, compa_down, key_bitlen, circ);
            }
        }
    }

    if (keys.size() > orig_n) {
        keys.resize(orig_n);
        for (auto& p : payloads) {
            p.wires.resize(orig_n);
        }
    }
}

static uint32_t make_composite_key(uint32_t key, uint32_t tag, uint32_t key_bits, BooleanCircuit* circ) {
    std::vector<uint32_t> key_bits_vec = circ->PutSplitterGate(key);
    key_bits_vec.insert(key_bits_vec.begin(), tag);
    return circ->PutCombinerGate(key_bits_vec);
}

static uint32_t make_row_key(uint32_t row_idx, uint32_t row_bits, uint32_t tag, BooleanCircuit* circ) {
    std::vector<uint32_t> row_bits_vec = circ->PutSplitterGate(row_idx);
    row_bits_vec.push_back(tag);
    return circ->PutCombinerGate(row_bits_vec);
}

static uint32_t mux_wire(uint32_t a, uint32_t b, uint32_t sel, BooleanCircuit* circ) {
    std::vector<uint32_t> out = circ->PutMUXGate({a}, {b}, sel);
    return out[0];
}

static std::vector<uint32_t> relabel_column(
    const std::vector<uint32_t>& ids,
    const std::vector<uint32_t>& map_key,
    const std::vector<uint32_t>& map_val,
    uint32_t id_bits,
    uint32_t row_bits,
    BooleanCircuit* circ) {

    size_t n_ids = ids.size();
    size_t n_map = map_key.size();
    std::vector<uint32_t> keys(n_map + n_ids);
    std::vector<uint32_t> tags(n_map + n_ids);
    std::vector<uint32_t> rows(n_map + n_ids);
    std::vector<uint32_t> vals(n_map + n_ids);

    share* tag0_s = circ->PutCONSGate(static_cast<uint64_t>(0), 1);
    share* tag1_s = circ->PutCONSGate(static_cast<uint64_t>(1), 1);
    uint32_t tag0 = tag0_s->get_wire_id(0);
    uint32_t tag1 = tag1_s->get_wire_id(0);

    share* zero_id_s = circ->PutCONSGate(static_cast<uint64_t>(0), id_bits);
    uint32_t zero_id = pack_yao_value(zero_id_s, id_bits, circ);

    share* zero_row_s = circ->PutCONSGate(static_cast<uint64_t>(0), row_bits);
    uint32_t zero_row = pack_yao_value(zero_row_s, row_bits, circ);

    for (size_t i = 0; i < n_map; ++i) {
        keys[i] = make_composite_key(map_key[i], tag0, id_bits, circ);
        tags[i] = tag0;
        rows[i] = zero_row;
        vals[i] = map_val[i];
    }
    for (size_t i = 0; i < n_ids; ++i) {
        size_t idx = n_map + i;
        keys[idx] = make_composite_key(ids[i], tag1, id_bits, circ);
        tags[idx] = tag1;
        share* row_s = circ->PutCONSGate(static_cast<uint64_t>(i), row_bits);
        rows[idx] = pack_yao_value(row_s, row_bits, circ);
        vals[idx] = zero_id;
    }

    std::vector<PayloadArray> payloads;
    payloads.push_back({vals, id_bits});
    payloads.push_back({tags, 1});
    payloads.push_back({rows, row_bits});
    bitonic_full_sort_multi(keys, payloads, id_bits + 1, circ);

    std::vector<uint32_t>& sorted_vals = payloads[0].wires;
    std::vector<uint32_t>& sorted_tags = payloads[1].wires;
    std::vector<uint32_t>& sorted_rows = payloads[2].wires;

    for (size_t i = 1; i < n_map + n_ids; ++i) {
        sorted_vals[i] = mux_wire(sorted_vals[i - 1], sorted_vals[i], sorted_tags[i], circ);
    }

    std::vector<uint32_t> row_keys(n_map + n_ids);
    share* one_s = circ->PutCONSGate(static_cast<uint64_t>(1), 1);
    uint32_t one = one_s->get_wire_id(0);
    for (size_t i = 0; i < n_map + n_ids; ++i) {
        uint32_t tag_inv = circ->PutXORGate(sorted_tags[i], one);
        row_keys[i] = make_row_key(sorted_rows[i], row_bits, tag_inv, circ);
    }

    std::vector<PayloadArray> row_payloads;
    row_payloads.push_back({sorted_vals, id_bits});
    bitonic_full_sort_multi(row_keys, row_payloads, row_bits + 1, circ);

    std::vector<uint32_t> out(n_ids);
    for (size_t i = 0; i < n_ids; ++i) {
        out[i] = row_payloads[0].wires[i];
    }
    return out;
}

static share* wrap_yao_wire(uint32_t wire, BooleanCircuit* circ) {
    std::vector<uint32_t> bits = circ->PutSplitterGate(wire);
    return new boolshare(bits, circ);
}

static void push_public_output_gates(
    const std::vector<uint32_t>& wires,
    BooleanCircuit* yc,
    std::vector<share*>* out_gates) {

    out_gates->reserve(out_gates->size() + wires.size());
    for (uint32_t wire : wires) {
        share* y = wrap_yao_wire(wire, yc);
        out_gates->push_back(yc->PutOUTGate(y, ALL));
    }
}

static void push_output_gates(
    const std::vector<uint32_t>& wires,
    uint32_t bitlen,
    ArithmeticCircuit* ac,
    BooleanCircuit* yc,
    BooleanCircuit* bc,
    std::vector<share*>* out_gates) {

    out_gates->reserve(out_gates->size() + wires.size());
    for (uint32_t wire : wires) {
        share* y = wrap_yao_wire(wire, yc);
        share* a = ac->PutY2AGate(y, bc);
        out_gates->push_back(ac->PutSharedOUTGate(a));
    }
}

int main(int argc, char** argv) {
    e_role role = SERVER;
    uint32_t int_role = 0;
    uint32_t int_port = 7766;
    uint32_t secparam = 128;
    uint32_t nthreads = 1;
    uint32_t layer = 0;
    uint32_t n = 0;
    uint32_t m = 0;
    uint32_t d = 0;
    uint32_t id_bits = 0;
    uint32_t vec_bits = 0;
    uint32_t x2_bits = 0;
    uint32_t down_bits = 0;
    uint32_t down_n = 0;
    uint32_t perm_bits = 32;
    uint32_t seed = 0;
    std::string address = "127.0.0.1";
    std::string in_dir;
    std::string out_dir;
    std::string permute_mode = "public";
    std::string map_in_key;
    std::string map_in_val;
    std::string entry_in;
    std::string entry_out;

    parsing_ctx options[] = {
        { (void*)&int_role, T_NUM, "r", "Role: 0/1 (required)", true, false },
        { (void*)&layer, T_NUM, "l", "Layer index", true, false },
        { (void*)&n, T_NUM, "n", "Layer size N", true, false },
        { (void*)&m, T_NUM, "m", "Layer M", true, false },
        { (void*)&d, T_NUM, "d", "Layer D", true, false },
        { (void*)&id_bits, T_NUM, "i", "ID bit-length", true, false },
        { (void*)&vec_bits, T_NUM, "v", "Vector bit-length", true, false },
        { (void*)&x2_bits, T_NUM, "x", "x2 bit-length", true, false },
        { (void*)&down_bits, T_NUM, "u", "Down bit-length (0 if none)", false, false },
        { (void*)&down_n, T_NUM, "U", "Down layer size (if map-in)", false, false },
        { (void*)&perm_bits, T_NUM, "k", "Permutation key bit-length", false, false },
        { (void*)&seed, T_NUM, "z", "RNG seed", false, false },
        { (void*)&secparam, T_NUM, "s", "Security parameter", false, false },
        { (void*)&nthreads, T_NUM, "t", "Threads", false, false },
        { (void*)&address, T_STR, "a", "Address", false, false },
        { (void*)&int_port, T_NUM, "p", "Port", false, false },
        { (void*)&in_dir, T_STR, "I", "Input share dir", true, false },
        { (void*)&out_dir, T_STR, "O", "Output share dir", true, false },
        { (void*)&permute_mode, T_STR, "P", "Permutation mode: public or secret", false, false },
        { (void*)&map_in_key, T_STR, "K", "Map-in key file", false, false },
        { (void*)&map_in_val, T_STR, "V", "Map-in val file", false, false },
        { (void*)&entry_in, T_STR, "E", "Entry-point input file", false, false },
        { (void*)&entry_out, T_STR, "F", "Entry-point output file", false, false },
    };

    if (!parse_options(&argc, &argv, options, sizeof(options) / sizeof(parsing_ctx))) {
        print_usage(argv[0], options, sizeof(options) / sizeof(parsing_ctx));
        return 0;
    }

    assert(int_role < 2);
    role = static_cast<e_role>(int_role);
    assert(int_port < (1U << 16));
    uint16_t port = static_cast<uint16_t>(int_port);

    if (permute_mode != "public" && permute_mode != "secret") {
        std::cerr << "permute_mode must be 'public' or 'secret' (got: " << permute_mode << ")\n";
        return 1;
    }
    bool permute_public = (permute_mode == "public");

    if (perm_bits > 32) {
        std::cerr << "perm_bits > 32 is not supported for 2PC reshuffle (perm_bits="
                  << perm_bits << "). Use --perm_bits <= 32.\n";
        return 1;
    }

    if (n == 0 || m == 0 || d == 0 || id_bits == 0 || vec_bits == 0 || x2_bits == 0) {
        std::cerr << "invalid layer config\n";
        return 1;
    }

    reshuffle_debug("reshuffle: parsed config");

    const size_t neigh_count = static_cast<size_t>(n) * m;
    const size_t vec_count = static_cast<size_t>(n) * d;

    auto neigh_shares = read_bin(in_dir + "/layer" + std::to_string(layer) + "_neigh.bin", neigh_count, id_bits);
    auto vec_shares = read_bin(in_dir + "/layer" + std::to_string(layer) + "_vecs.bin", vec_count, vec_bits);
    auto x2_shares = read_bin(in_dir + "/layer" + std::to_string(layer) + "_x2.bin", n, x2_bits);
    auto dummy_shares = read_bin(in_dir + "/layer" + std::to_string(layer) + "_is_dummy.bin", n, 1);

    std::vector<uint64_t> down_shares;
    bool has_down = (down_bits != 0);
    if (has_down) {
        down_shares = read_bin(in_dir + "/layer" + std::to_string(layer) + "_down.bin", n, down_bits);
    }

    std::vector<uint64_t> map_in_key_shares;
    std::vector<uint64_t> map_in_val_shares;
    bool has_map_in = !map_in_key.empty() && !map_in_val.empty();
    if (has_map_in && !has_down) {
        std::cerr << "map_in requires down_bits\n";
        return 1;
    }
    if (has_map_in && down_n == 0) {
        std::cerr << "map_in requires down_n\n";
        return 1;
    }
    if (has_map_in) {
        map_in_key_shares = read_bin(map_in_key, down_n, down_bits);
        map_in_val_shares = read_bin(map_in_val, down_n, down_bits);
    }

    uint64_t entry_share = 0;
    bool has_entry = !entry_in.empty() && !entry_out.empty();
    if (has_entry) {
        auto entry_vec = read_bin(entry_in, 1, id_bits);
        entry_share = entry_vec[0];
    }

    reshuffle_debug("reshuffle: loaded input shares");

    ABYParty party(role, address, port, LT, secparam, nthreads, MT_OT);
    auto sharings = party.GetSharings();
    auto* ac = static_cast<ArithmeticCircuit*>(sharings[S_ARITH]->GetCircuitBuildRoutine());
    auto* bc = static_cast<BooleanCircuit*>(sharings[S_BOOL]->GetCircuitBuildRoutine());
    auto* yc = static_cast<BooleanCircuit*>(sharings[S_YAO]->GetCircuitBuildRoutine());

    reshuffle_debug("reshuffle: initialized ABY circuits");

    std::vector<uint32_t> neigh_y(neigh_count);
    std::vector<uint32_t> vec_y(vec_count);
    std::vector<uint32_t> x2_y(n);
    std::vector<uint32_t> dummy_y(n);
    std::vector<uint32_t> down_y;
    std::vector<uint32_t> map_in_key_y;
    std::vector<uint32_t> map_in_val_y;

    for (size_t i = 0; i < neigh_count; ++i) {
        share* a = ac->PutSharedINGate(neigh_shares[i], id_bits);
        share* y = yc->PutA2YGate(a);
        neigh_y[i] = pack_yao_value(y, id_bits, yc);
    }
    for (size_t i = 0; i < vec_count; ++i) {
        share* a = ac->PutSharedINGate(vec_shares[i], vec_bits);
        share* y = yc->PutA2YGate(a);
        vec_y[i] = pack_yao_value(y, vec_bits, yc);
    }
    for (size_t i = 0; i < n; ++i) {
        share* a = ac->PutSharedINGate(x2_shares[i], x2_bits);
        share* y = yc->PutA2YGate(a);
        x2_y[i] = pack_yao_value(y, x2_bits, yc);
    }
    for (size_t i = 0; i < n; ++i) {
        share* a = ac->PutSharedINGate(dummy_shares[i], 1);
        share* y = yc->PutA2YGate(a);
        dummy_y[i] = pack_yao_value(y, 1, yc);
    }
    if (has_down) {
        down_y.resize(n);
        for (size_t i = 0; i < n; ++i) {
            share* a = ac->PutSharedINGate(down_shares[i], down_bits);
            share* y = yc->PutA2YGate(a);
            down_y[i] = pack_yao_value(y, down_bits, yc);
        }
    }
    if (has_map_in) {
        map_in_key_y.resize(n);
        map_in_val_y.resize(n);
        for (size_t i = 0; i < n; ++i) {
            share* a_k = ac->PutSharedINGate(map_in_key_shares[i], down_bits);
            share* y_k = yc->PutA2YGate(a_k);
            map_in_key_y[i] = pack_yao_value(y_k, down_bits, yc);

            share* a_v = ac->PutSharedINGate(map_in_val_shares[i], down_bits);
            share* y_v = yc->PutA2YGate(a_v);
            map_in_val_y[i] = pack_yao_value(y_v, down_bits, yc);
        }
    }

    reshuffle_debug("reshuffle: converted inputs to Yao wires");

    std::vector<uint32_t> old_id_y(n);
    for (size_t i = 0; i < n; ++i) {
        share* c = yc->PutCONSGate(static_cast<uint64_t>(i), id_bits);
        old_id_y[i] = pack_yao_value(c, id_bits, yc);
    }

    reshuffle_debug("reshuffle: built old_id wires");

    std::vector<uint32_t> perm_key_y(n);
    std::mt19937_64 rng(seed + (role == SERVER ? 0 : 1337));
    uint64_t max_key = (perm_bits >= 64) ? ~0ULL : ((1ULL << perm_bits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(0, max_key);
    for (size_t i = 0; i < n; ++i) {
        uint64_t share_val = dist(rng);
        share* a = ac->PutSharedINGate(share_val, perm_bits);
        share* y = yc->PutA2YGate(a);
        perm_key_y[i] = pack_yao_value(y, perm_bits, yc);
    }

    reshuffle_debug("reshuffle: built perm keys");

    std::vector<std::vector<uint32_t>> neigh_cols(m, std::vector<uint32_t>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            neigh_cols[j][i] = neigh_y[i * m + j];
        }
    }
    reshuffle_debug("reshuffle: arranged neighbor columns");

    std::vector<std::vector<uint32_t>> vec_cols(d, std::vector<uint32_t>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            vec_cols[j][i] = vec_y[i * d + j];
        }
    }
    reshuffle_debug("reshuffle: arranged vector columns");

    std::vector<PayloadArray> payloads;
    payloads.push_back({old_id_y, id_bits});
    if (permute_public) {
        payloads.push_back({dummy_y, 1});
    } else {
        for (size_t j = 0; j < m; ++j) {
            payloads.push_back({neigh_cols[j], id_bits});
        }
        for (size_t j = 0; j < d; ++j) {
            payloads.push_back({vec_cols[j], vec_bits});
        }
        payloads.push_back({x2_y, x2_bits});
        payloads.push_back({dummy_y, 1});
        if (has_down) {
            payloads.push_back({down_y, down_bits});
        }
    }

    reshuffle_debug("reshuffle: sorting payloads");
    bitonic_full_sort_multi(perm_key_y, payloads, perm_bits, yc);
    reshuffle_debug("reshuffle: sorting payloads done");

    size_t idx = 0;
    old_id_y = std::move(payloads[idx++].wires);
    if (permute_public) {
        dummy_y = std::move(payloads[idx++].wires);
    } else {
        for (size_t j = 0; j < m; ++j) {
            neigh_cols[j] = std::move(payloads[idx++].wires);
        }
        for (size_t j = 0; j < d; ++j) {
            vec_cols[j] = std::move(payloads[idx++].wires);
        }
        x2_y = std::move(payloads[idx++].wires);
        dummy_y = std::move(payloads[idx++].wires);
        if (has_down) {
            down_y = std::move(payloads[idx++].wires);
        }
    }

    std::vector<uint32_t> map_key_y = old_id_y;
    std::vector<uint32_t> map_val_y(n);
    for (size_t i = 0; i < n; ++i) {
        share* c = yc->PutCONSGate(static_cast<uint64_t>(i), id_bits);
        map_val_y[i] = pack_yao_value(c, id_bits, yc);
    }

    uint32_t row_bits = (n <= 1) ? 1 : (floor_log2(n - 1) + 1);
    for (size_t j = 0; j < m; ++j) {
        neigh_cols[j] = relabel_column(neigh_cols[j], map_key_y, map_val_y, id_bits, row_bits, yc);
    }
    if (has_down && has_map_in) {
        down_y = relabel_column(down_y, map_in_key_y, map_in_val_y, down_bits, row_bits, yc);
    }

    reshuffle_debug("reshuffle: relabel completed");

    share* zero_id_s = yc->PutCONSGate(static_cast<uint64_t>(0), id_bits);
    uint32_t dummy_id_wire = pack_yao_value(zero_id_s, id_bits, yc);
    share* seen_s = yc->PutCONSGate(static_cast<uint64_t>(0), 1);
    uint32_t seen_wire = seen_s->get_wire_id(0);
    share* one_s = yc->PutCONSGate(static_cast<uint64_t>(1), 1);
    uint32_t one_wire = one_s->get_wire_id(0);

    for (size_t i = 0; i < n; ++i) {
        uint32_t not_seen = yc->PutXORGate(seen_wire, one_wire);
        uint32_t take = yc->PutANDGate(dummy_y[i], not_seen);
        share* idx_s = yc->PutCONSGate(static_cast<uint64_t>(i), id_bits);
        uint32_t idx_wire = pack_yao_value(idx_s, id_bits, yc);
        dummy_id_wire = mux_wire(idx_wire, dummy_id_wire, take, yc);
        seen_wire = yc->PutORGate(seen_wire, dummy_y[i]);
    }

    uint32_t entry_out_wire = 0;
    if (has_entry) {
        share* entry_a = ac->PutSharedINGate(entry_share, id_bits);
        share* entry_y = yc->PutA2YGate(entry_a);
        uint32_t entry_wire = pack_yao_value(entry_y, id_bits, yc);
        std::vector<uint32_t> entry_bits = yc->PutSplitterGate(entry_wire);

        share* zero = yc->PutCONSGate(static_cast<uint64_t>(0), id_bits);
        entry_out_wire = pack_yao_value(zero, id_bits, yc);

        for (size_t i = 0; i < n; ++i) {
            std::vector<uint32_t> key_bits = yc->PutSplitterGate(map_key_y[i]);
            uint32_t eq = yc->PutEQGate(entry_bits, key_bits);
            entry_out_wire = mux_wire(map_val_y[i], entry_out_wire, eq, yc);
        }
    }

    std::vector<uint32_t> neigh_out(neigh_count);
    std::vector<uint32_t> vec_out(vec_count);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            neigh_out[i * m + j] = neigh_cols[j][i];
        }
        for (size_t j = 0; j < d; ++j) {
            vec_out[i * d + j] = vec_cols[j][i];
        }
    }

    std::vector<share*> neigh_out_gates;
    std::vector<share*> vec_out_gates;
    std::vector<share*> x2_out_gates;
    std::vector<share*> dummy_out_gates;
    share* dummy_id_out_gate = nullptr;
    std::vector<share*> down_out_gates;
    std::vector<share*> map_key_out_gates;
    std::vector<share*> map_val_out_gates;
    std::vector<share*> perm_out_gates;
    std::vector<share*> entry_out_gates;

    push_output_gates(neigh_out, id_bits, ac, yc, bc, &neigh_out_gates);
    push_output_gates(vec_out, vec_bits, ac, yc, bc, &vec_out_gates);
    push_output_gates(x2_y, x2_bits, ac, yc, bc, &x2_out_gates);
    push_output_gates(dummy_y, 1, ac, yc, bc, &dummy_out_gates);
    dummy_id_out_gate = yc->PutOUTGate(wrap_yao_wire(dummy_id_wire, yc), ALL);
    if (has_down) {
        push_output_gates(down_y, down_bits, ac, yc, bc, &down_out_gates);
    }
    push_output_gates(map_key_y, id_bits, ac, yc, bc, &map_key_out_gates);
    push_output_gates(map_val_y, id_bits, ac, yc, bc, &map_val_out_gates);
    if (permute_public) {
        push_public_output_gates(old_id_y, yc, &perm_out_gates);
    }
    if (has_entry) {
        push_output_gates({entry_out_wire}, id_bits, ac, yc, bc, &entry_out_gates);
    }

    reshuffle_debug("reshuffle: executing circuit");
    party.ExecCircuit();
    reshuffle_debug("reshuffle: circuit executed");

    std::vector<uint64_t> neigh_out_shares;
    std::vector<uint64_t> vec_out_shares;
    std::vector<uint64_t> x2_out_shares;
    std::vector<uint64_t> dummy_out_shares;
    std::vector<uint64_t> dummy_id_out_shares;
    std::vector<uint64_t> down_out_shares;
    std::vector<uint64_t> map_key_out_shares;
    std::vector<uint64_t> map_val_out_shares;
    std::vector<uint64_t> perm_out_shares;
    std::vector<uint64_t> entry_out_shares;
    neigh_out_shares.resize(neigh_out_gates.size());
    for (size_t i = 0; i < neigh_out_gates.size(); ++i) {
        neigh_out_shares[i] = neigh_out_gates[i]->get_clear_value<uint64_t>();
    }
    vec_out_shares.resize(vec_out_gates.size());
    for (size_t i = 0; i < vec_out_gates.size(); ++i) {
        vec_out_shares[i] = vec_out_gates[i]->get_clear_value<uint64_t>();
    }
    x2_out_shares.resize(x2_out_gates.size());
    for (size_t i = 0; i < x2_out_gates.size(); ++i) {
        x2_out_shares[i] = x2_out_gates[i]->get_clear_value<uint64_t>();
    }
    dummy_out_shares.resize(dummy_out_gates.size());
    for (size_t i = 0; i < dummy_out_gates.size(); ++i) {
        dummy_out_shares[i] = dummy_out_gates[i]->get_clear_value<uint64_t>();
    }
    dummy_id_out_shares.resize(1);
    dummy_id_out_shares[0] = dummy_id_out_gate->get_clear_value<uint64_t>();
    if (has_down) {
        down_out_shares.resize(down_out_gates.size());
        for (size_t i = 0; i < down_out_gates.size(); ++i) {
            down_out_shares[i] = down_out_gates[i]->get_clear_value<uint64_t>();
        }
    }
    map_key_out_shares.resize(map_key_out_gates.size());
    for (size_t i = 0; i < map_key_out_gates.size(); ++i) {
        map_key_out_shares[i] = map_key_out_gates[i]->get_clear_value<uint64_t>();
    }
    map_val_out_shares.resize(map_val_out_gates.size());
    for (size_t i = 0; i < map_val_out_gates.size(); ++i) {
        map_val_out_shares[i] = map_val_out_gates[i]->get_clear_value<uint64_t>();
    }
    if (permute_public) {
        perm_out_shares.resize(perm_out_gates.size());
        for (size_t i = 0; i < perm_out_gates.size(); ++i) {
            perm_out_shares[i] = perm_out_gates[i]->get_clear_value<uint64_t>();
        }
    }
    if (has_entry) {
        entry_out_shares.resize(entry_out_gates.size());
        for (size_t i = 0; i < entry_out_gates.size(); ++i) {
            entry_out_shares[i] = entry_out_gates[i]->get_clear_value<uint64_t>();
        }
    }

    if (permute_public) {
        if (perm_out_shares.size() != n) {
            std::cerr << "perm_out size mismatch: expected " << n
                      << " got " << perm_out_shares.size() << "\n";
            return 1;
        }
        neigh_out_shares = apply_perm_rows(neigh_out_shares, perm_out_shares, m);
        vec_out_shares = apply_perm_rows(vec_out_shares, perm_out_shares, d);
        x2_out_shares = apply_perm_1d(x2_out_shares, perm_out_shares);
        if (has_down) {
            down_out_shares = apply_perm_1d(down_out_shares, perm_out_shares);
        }
    }

    std::filesystem::create_directories(out_dir);
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_neigh.bin", neigh_out_shares, id_bits);
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_vecs.bin", vec_out_shares, vec_bits);
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_x2.bin", x2_out_shares, x2_bits);
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_is_dummy.bin", dummy_out_shares, 1);
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_dummy_id.bin",
              dummy_id_out_shares, id_bits);
    if (has_down) {
        write_bin(out_dir + "/layer" + std::to_string(layer) + "_down.bin", down_out_shares, down_bits);
    }
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_map_key.bin", map_key_out_shares, id_bits);
    write_bin(out_dir + "/layer" + std::to_string(layer) + "_map_val.bin", map_val_out_shares, id_bits);
    if (permute_public) {
        write_bin(out_dir + "/layer" + std::to_string(layer) + "_perm.bin", perm_out_shares, id_bits);
    }
    if (has_entry) {
        write_bin(entry_out, entry_out_shares, id_bits);
    }

    return 0;
}
