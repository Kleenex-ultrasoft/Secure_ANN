#ifndef HNSECW_BATCH_B2Y_H_
#define HNSECW_BATCH_B2Y_H_

#include "../../abycore/aby/abyparty.h"
#include <string>

// Batch search of one HNSW layer for num_queries queries in parallel
// (Algorithm 4).  Returns 0 on success, 2 if an opened value disagrees with
// the plaintext reference.
int32_t hnsecw_batch_b2y(
    e_role role, const std::string& address, uint16_t port, seclvl seclvl,
    uint32_t num_queries, uint32_t M, uint32_t D, uint32_t L_C, uint32_t L_W,
    uint32_t bitlen, uint32_t id_bitlen,
    uint32_t entry_id, const std::string& entry_file, const std::string& entry_out,
    uint32_t verbose, uint32_t nthreads, e_mt_gen_alg mt_alg);

#endif  // HNSECW_BATCH_B2Y_H_
