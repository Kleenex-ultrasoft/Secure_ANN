#ifndef HNSECW_SINGLE_B2Y_H_
#define HNSECW_SINGLE_B2Y_H_

#include "../../abycore/aby/abyparty.h"
#include <string>

int32_t hnsecw_single_b2y(
    e_role role,
    const std::string& address,
    uint16_t port,
    seclvl seclvl,
    uint32_t M,
    uint32_t D,
    uint32_t L_C,
    uint32_t L_W,
    uint32_t LV,
    uint32_t bitlen,
    uint32_t id_bitlen,
    uint32_t entry_id,
    const std::string& entry_file,
    const std::string& entry_out,
    uint32_t dummy_id_override,
    uint32_t debug_tag,
    uint32_t nthreads,
    e_mt_gen_alg mt_alg
);

#endif  // HNSECW_SINGLE_B2Y_H_
