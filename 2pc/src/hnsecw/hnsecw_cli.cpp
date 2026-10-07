// Command-line front end of the HNSecW two-party programs.
//
//   hnsecw_cli -r <0|1> -X <single|multi|batch> -m M -d D -c T -w ef -i id_bits [...]
//
// Start one process with -r 0 (server) and one with -r 1 (client).  One run
// searches one HNSW layer; bench/run_search.py chains the layers.

#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>

#include "../../abycore/aby/abyparty.h"
#include "hnsecw_batch_b2y.h"
#include "hnsecw_multi_b2y.h"
#include "hnsecw_single_b2y.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    uint32_t role = 0;
    std::string mode = "single";
    uint32_t num_queries = 1;
    uint32_t M = 8;
    uint32_t D = 10;
    uint32_t L_C = 4;
    uint32_t L_W = 4;
    uint32_t bitlen = 8;
    uint32_t id_bitlen = 8;
    uint32_t entry_id = std::numeric_limits<uint32_t>::max();
    std::string entry_file;
    std::string entry_out;
    uint32_t verbose = 0;
    uint32_t secparam = 128;
    uint32_t nthreads = 1;
    std::string address = "127.0.0.1";
    uint32_t port = 7766;

    parsing_ctx options[] = {
        { (void*) &role, T_NUM, "r", "Role: 0 (server) or 1 (client)", true, false },
        { (void*) &mode, T_STR, "X", "Mode: single|multi|batch (default: single)", false, false },
        { (void*) &num_queries, T_NUM, "q", "Number of queries, multi/batch (default: 1)", false, false },
        { (void*) &M, T_NUM, "m", "Degree bound M (default: 8)", false, false },
        { (void*) &D, T_NUM, "d", "Dimension (default: 10)", false, false },
        { (void*) &L_C, T_NUM, "c", "Iterations T = ef + Delta (default: 4)", false, false },
        { (void*) &L_W, T_NUM, "w", "Result array size ef (default: 4)", false, false },
        { (void*) &bitlen, T_NUM, "b", "Bits per coordinate (default: 8)", false, false },
        { (void*) &id_bitlen, T_NUM, "i", "Bits per address (default: 8)", false, false },
        { (void*) &entry_id, T_NUM, "e", "Entry point id (optional)", false, false },
        { (void*) &entry_file, T_STR, "f", "File with entry point ids, uint32 each (optional)", false, false },
        { (void*) &entry_out, T_STR, "o", "File the client writes the result ids to (optional)", false, false },
        { (void*) &verbose, T_NUM, "g", "Per-iteration output and open the distances in W (0/1)", false, false },
        { (void*) &secparam, T_NUM, "s", "Symmetric security parameter (default: 128)", false, false },
        { (void*) &nthreads, T_NUM, "t", "Threads for the setup phase (default: 1)", false, false },
        { (void*) &address, T_STR, "a", "Address of the server (default: 127.0.0.1)", false, false },
        { (void*) &port, T_NUM, "p", "Port (default: 7766)", false, false },
    };
    const uint32_t noptions = sizeof(options) / sizeof(parsing_ctx);

    if (!parse_options(&argc, &argv, options, noptions)) {
        print_usage(argv[0], options, noptions);
        std::cout << "\nExample:\n  " << argv[0]
                  << " -r 0 -X single -m 128 -d 128 -c 47 -w 20 -i 20\n\n";
        return 1;
    }
    if (role > 1 || port >= (1U << 16)) {
        std::cerr << "role must be 0 or 1 and the port below 65536" << std::endl;
        return 1;
    }

    const e_role r = (e_role) role;
    const seclvl lvl = get_sec_lvl(secparam);
    const e_mt_gen_alg mt_alg = MT_OT;

    if (mode == "single") {
        return hnsecw_single_b2y(r, address, (uint16_t) port, lvl, M, D, L_C, L_W, bitlen, id_bitlen,
                                 entry_id, entry_file, entry_out, verbose, nthreads, mt_alg);
    }
    if (mode == "multi") {
        return hnsecw_multi_b2y(r, address, (uint16_t) port, lvl, num_queries, M, D, L_C, L_W,
                                bitlen, id_bitlen, entry_id, entry_file, entry_out, verbose,
                                nthreads, mt_alg);
    }
    if (mode == "batch") {
        return hnsecw_batch_b2y(r, address, (uint16_t) port, lvl, num_queries, M, D, L_C, L_W,
                                bitlen, id_bitlen, entry_id, entry_file, entry_out, verbose,
                                nthreads, mt_alg);
    }
    std::cerr << "unknown mode: " << mode << std::endl;
    return 1;
}
