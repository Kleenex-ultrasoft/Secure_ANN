#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include "../../abycore/aby/abyparty.h"
#include "hnsecw_batch_b2y.h"
#include "hnsecw_multi_b2y_dyn.h"
#include "hnsecw_single_b2a.h"
#include "hnsecw_single_b2y.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <limits>
#include <string>
#include <vector>

static bool read_json_uint32(const std::string& content, const std::string& key, uint32_t* out) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (std::regex_search(content, match, pattern) && match.size() >= 2) {
        *out = static_cast<uint32_t>(std::stoul(match[1].str()));
        return true;
    }
    return false;
}

static bool read_json_string(const std::string& content, const std::string& key, std::string* out) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(content, match, pattern) && match.size() >= 2) {
        *out = match[1].str();
        return true;
    }
    return false;
}

static void apply_config(
    const std::string& path,
    std::string* mode,
    std::string* protocol,
    uint32_t* entry_id,
    std::string* entry_file,
    std::string* entry_out,
    uint32_t* num_queries,
    uint32_t* M,
    uint32_t* D,
    uint32_t* L_C,
    uint32_t* L_W,
    uint32_t* LV,
    uint32_t* bitlen,
    uint32_t* id_bitlen,
    uint32_t* yao_eq_thresh,
    uint32_t* yao_vg_thresh,
    uint32_t* yao_vd_thresh,
    uint32_t* yao_dedup_thresh,
    std::string* dedup_algo,
    uint32_t* force_dedup_yao,
    uint32_t* dummy_id,
    uint32_t* debug_tag,
    uint32_t* secparam,
    uint32_t* nthreads,
    std::string* address,
    uint16_t* port) {

    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "failed to open config: " << path << "\n";
        return;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    read_json_string(content, "mode", mode);
    read_json_string(content, "protocol", protocol);
    read_json_string(content, "address", address);
    read_json_string(content, "entry_file", entry_file);
    read_json_string(content, "entry_out", entry_out);

    read_json_uint32(content, "entry_id", entry_id);
    read_json_uint32(content, "num_queries", num_queries);
    read_json_uint32(content, "M", M);
    read_json_uint32(content, "D", D);
    read_json_uint32(content, "L_C", L_C);
    read_json_uint32(content, "L_W", L_W);
    read_json_uint32(content, "LV", LV);
    read_json_uint32(content, "bitlen", bitlen);
    read_json_uint32(content, "id_bitlen", id_bitlen);
    read_json_uint32(content, "yao_eq_thresh", yao_eq_thresh);
    read_json_uint32(content, "yao_vg_thresh", yao_vg_thresh);
    read_json_uint32(content, "yao_vd_thresh", yao_vd_thresh);
    read_json_uint32(content, "yao_dedup_thresh", yao_dedup_thresh);
    read_json_string(content, "dedup_algo", dedup_algo);
    read_json_uint32(content, "force_dedup_yao", force_dedup_yao);
    read_json_uint32(content, "dummy_id", dummy_id);
    read_json_uint32(content, "debug_tag", debug_tag);
    read_json_uint32(content, "secparam", secparam);
    read_json_uint32(content, "nthreads", nthreads);

    uint32_t port_val = 0;
    if (read_json_uint32(content, "port", &port_val) && port_val < (1U << 16)) {
        *port = static_cast<uint16_t>(port_val);
    }
}

static int32_t read_options(
    int32_t* argcp,
    char** argvp,
    e_role* role,
    std::string* mode,
    std::string* protocol,
    uint32_t* entry_id,
    std::string* entry_file,
    std::string* entry_out,
    uint32_t* num_queries,
    uint32_t* M,
    uint32_t* D,
    uint32_t* L_C,
    uint32_t* L_W,
    uint32_t* LV,
    uint32_t* bitlen,
    uint32_t* id_bitlen,
    uint32_t* yao_eq_thresh,
    uint32_t* yao_vg_thresh,
    uint32_t* yao_vd_thresh,
    uint32_t* yao_dedup_thresh,
    std::string* dedup_algo,
    uint32_t* force_dedup_yao,
    uint32_t* dummy_id,
    uint32_t* debug_tag,
    uint32_t* secparam,
    uint32_t* nthreads,
    std::string* address,
    uint16_t* port) {

    uint32_t int_role = 0;
    uint32_t int_port = 0;

    parsing_ctx options[] = {
        { (void*) &int_role, T_NUM, "r", "Role: 0/1 (required)", true, false },
        { (void*) mode, T_STR, "X", "Mode: single|multi|batch (default: single)", false, false },
        { (void*) protocol, T_STR, "P", "Protocol: b2y|b2a|dyn (default: b2y)", false, false },
        { (void*) entry_id, T_NUM, "e", "Entry ID override (optional)", false, false },
        { (void*) entry_file, T_STR, "f", "Entry ID file (optional)", false, false },
        { (void*) entry_out, T_STR, "o", "Entry output file (optional)", false, false },
        { (void*) num_queries, T_NUM, "q", "Number of queries (default: 1)", false, false },
        { (void*) M, T_NUM, "m", "Neighbors per expansion M (default: 8)", false, false },
        { (void*) D, T_NUM, "d", "Dimension D (default: 10)", false, false },
        { (void*) L_C, T_NUM, "c", "Length of C vector (default: 4)", false, false },
        { (void*) L_W, T_NUM, "w", "Length of W vector (default: 4)", false, false },
        { (void*) LV, T_NUM, "v", "Visited multiplier LV (default: 0 => L_C)", false, false },
        { (void*) bitlen, T_NUM, "b", "Bit-length (default: 8)", false, false },
        { (void*) id_bitlen, T_NUM, "i", "ID bit-length (default: 8)", false, false },
        { (void*) yao_eq_thresh, T_NUM, "y", "Yao EQ threshold (dyn/multi)", false, false },
        { (void*) yao_vg_thresh, T_NUM, "G", "Yao VG threshold (dyn/multi)", false, false },
        { (void*) yao_vd_thresh, T_NUM, "V", "Yao VD threshold (dyn/multi)", false, false },
        { (void*) yao_dedup_thresh, T_NUM, "Y", "Yao dedup threshold (batch)", false, false },
        { (void*) dedup_algo, T_STR, "A", "Batch dedup algo: bitonic|radix (default: bitonic)", false, false },
        { (void*) force_dedup_yao, T_NUM, "F", "Force Yao dedup (0/1, batch)", false, false },
        { (void*) dummy_id, T_NUM, "U", "Dummy ID override (optional)", false, false },
        { (void*) debug_tag, T_NUM, "g", "Enable debug tag payload (0/1, default: 0)", false, false },
        { (void*) secparam, T_NUM, "s", "Security parameter (default: 128)", false, false },
        { (void*) nthreads, T_NUM, "t", "ABY worker threads (default: 1)", false, false },
        { (void*) address, T_STR, "a", "IP address (default: 127.0.0.1)", false, false },
        { (void*) &int_port, T_NUM, "p", "Port (default: 7766)", false, false }
    };

    if (!parse_options(argcp, &argvp, options, sizeof(options) / sizeof(parsing_ctx))) {
        print_usage(argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
        std::cout << "\nExample usage:\n  " << argvp[0]
                  << " -r 0 -X single -P b2y -m 8 -d 10 -c 4 -w 4 -b 8 -i 8\n\n";
        exit(0);
    }

    assert(int_role < 2);
    *role = (e_role) int_role;
    if (int_port != 0) {
        assert(int_port < 1 << (sizeof(uint16_t) * 8));
        *port = (uint16_t) int_port;
    }

    return 1;
}

int main(int argc, char** argv) {
    e_role role;
    std::string mode = "single";
    std::string protocol = "b2y";
    std::string config_path;
    uint32_t entry_id = std::numeric_limits<uint32_t>::max();
    std::string entry_file;
    std::string entry_out;
    uint32_t num_queries = 1;
    uint32_t M = 8;
    uint32_t D = 10;
    uint32_t L_C = 4;
    uint32_t L_W = 4;
    uint32_t LV = 0;
    uint32_t bitlen = 8;
    uint32_t id_bitlen = 8;
    uint32_t yao_eq_thresh = 0;
    uint32_t yao_vg_thresh = 0;
    uint32_t yao_vd_thresh = 0;
    uint32_t yao_dedup_thresh = 0;
    std::string dedup_algo = "bitonic";
    uint32_t force_dedup_yao = 0;
    uint32_t dummy_id = std::numeric_limits<uint32_t>::max();
    uint32_t debug_tag = 0;
    uint32_t secparam = 128;
    uint32_t nthreads = 1;
    uint16_t port = 7766;
    std::string address = "127.0.0.1";
    e_mt_gen_alg mt_alg = MT_OT;

    std::vector<std::string> arg_store;
    arg_store.reserve(argc + 8);
    arg_store.emplace_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            arg_store.emplace_back("-X");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--protocol" && i + 1 < argc) {
            arg_store.emplace_back("-P");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--entry_id" && i + 1 < argc) {
            arg_store.emplace_back("-e");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--entry_file" && i + 1 < argc) {
            arg_store.emplace_back("-f");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--entry_out" && i + 1 < argc) {
            arg_store.emplace_back("-o");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--dummy_id" && i + 1 < argc) {
            arg_store.emplace_back("-U");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            continue;
        }
        if (arg == "--dedup_algo" && i + 1 < argc) {
            arg_store.emplace_back("-A");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--dedup_force_yao" && i + 1 < argc) {
            arg_store.emplace_back("-F");
            arg_store.emplace_back(argv[++i]);
            continue;
        }
        arg_store.emplace_back(arg);
    }

    std::vector<char*> argv_clean;
    argv_clean.reserve(arg_store.size());
    for (auto& s : arg_store) {
        argv_clean.push_back(const_cast<char*>(s.c_str()));
    }
    int32_t argc_clean = static_cast<int32_t>(argv_clean.size());

    if (!config_path.empty()) {
        apply_config(config_path, &mode, &protocol,
                     &entry_id, &entry_file, &entry_out,
                     &num_queries,
                     &M, &D, &L_C, &L_W, &LV, &bitlen, &id_bitlen,
                     &yao_eq_thresh, &yao_vg_thresh, &yao_vd_thresh,
                     &yao_dedup_thresh, &dedup_algo, &force_dedup_yao,
                     &dummy_id, &debug_tag, &secparam, &nthreads,
                     &address, &port);
    }

    char** argv_ptr = argv_clean.data();
    read_options(&argc_clean, argv_ptr, &role, &mode, &protocol,
                 &entry_id, &entry_file, &entry_out,
                 &num_queries,
                 &M, &D, &L_C, &L_W, &LV, &bitlen, &id_bitlen,
                 &yao_eq_thresh, &yao_vg_thresh, &yao_vd_thresh,
                 &yao_dedup_thresh, &dedup_algo, &force_dedup_yao,
                 &dummy_id, &debug_tag, &secparam, &nthreads,
                 &address, &port);

    if (LV == 0) {
        LV = L_C;
    }

    seclvl seclvl = get_sec_lvl(secparam);

    if (mode == "single") {
        if (protocol == "b2a") {
            return hnsecw_single_b2a(role, address, port, seclvl,
                                     M, D, L_C, L_W, LV, bitlen, id_bitlen,
                                     entry_id, entry_file, entry_out,
                                     dummy_id, debug_tag, nthreads, mt_alg);
        }
        return hnsecw_single_b2y(role, address, port, seclvl,
                                 M, D, L_C, L_W, LV, bitlen, id_bitlen,
                                 entry_id, entry_file, entry_out,
                                 dummy_id, debug_tag, nthreads, mt_alg);
    }

    if (mode == "multi") {
        if (num_queries == 0) {
            std::cerr << "num_queries must be > 0 for multi mode\n";
            return 1;
        }
        return hnsecw_multi_b2y_dyn(role, address, port, seclvl,
                                    num_queries, M, D, L_C, L_W, LV,
                                    bitlen, id_bitlen, yao_eq_thresh,
                                    yao_vg_thresh, yao_vd_thresh,
                                    entry_id, entry_file, entry_out,
                                    dummy_id, debug_tag,
                                    nthreads, mt_alg);
    }

    if (mode == "batch") {
        if (num_queries == 0) {
            std::cerr << "num_queries must be > 0 for batch mode\n";
            return 1;
        }
        uint32_t dedup_algo_id = 0;
        if (!dedup_algo.empty() && dedup_algo != "bitonic" && dedup_algo != "radix") {
            std::cerr << "unknown dedup_algo: " << dedup_algo << " (use bitonic|radix)\n";
            return 1;
        }
        if (dedup_algo == "radix") {
            dedup_algo_id = 1;
        }
        return hnsecw_batch_b2y(role, address, port, seclvl,
                                num_queries, M, D, L_C, L_W, LV,
                                bitlen, id_bitlen, yao_dedup_thresh,
                                dedup_algo_id, force_dedup_yao,
                                entry_id, entry_file, entry_out,
                                dummy_id,
                                debug_tag, nthreads, mt_alg);
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
