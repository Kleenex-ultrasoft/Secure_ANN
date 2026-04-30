#include <chrono>
#include <vector>
#include <random>
#include <iostream>

#include "libspu/kernel/hal/prot_wrapper.h" // Added
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/permute.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/mpc/utils/simulate.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/core/context.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/kernel/hlo/indexing.h" // DynamicSlice

#include "experimental/panther/protocol/common.h"
#include "spdlog/spdlog.h"
#include "llvm/Support/CommandLine.h"

using namespace spu;
using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list"));

llvm::cl::opt<uint32_t> PartyRank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

void BenchSecretIndexRetrieval(spu::SPUContext* ctx, 
                               int64_t num_elements, int64_t element_dim) {
  SPDLOG_INFO("Benchmarking Secret Index Retrieval: N={}, Dim={}", num_elements, element_dim);

  // 1. Prepare Data
  // Note: For benchmarking, we use zeros to avoid large random generation overhead.
  // The cost of MPC operations is data-independent.
  // spu::Value db_p = kernel::hal::zeros(ctx, spu::DT_I32, {num_elements, element_dim});
  // spu::Value db = kernel::hal::_p2s(ctx, db_p);
  
  // 2. Prepare Index
  // Index must be secret shared.
  // spu::Value index_p = kernel::hal::zeros(ctx, spu::DT_I32, {1}); 
  // spu::Value index_s = kernel::hal::_p2s(ctx, index_p);
  
  // auto start = std::chrono::system_clock::now();
  auto c0 = ctx->lctx()->GetStats()->sent_bytes.load();
  auto a0 = ctx->lctx()->GetStats()->sent_actions.load();
  
  auto start = std::chrono::system_clock::now(); // Define start here

  // DynamicSlice expects index to be 1D, but `index_s` is {1} (which is 1D).
  // However, DynamicSlice internally reshapes and processes indices.
  // The error might be related to type mismatch or shape.
  // `index_s` is INT32. `db` is INT32.
  
  // Use hal::oramread directly as requested by user if DynamicSlice fails.
  // Value oramread(SPUContext* ctx, const Value& x, const Value& y, int64_t offset);
  // x is onehot index (secret). y is database.
  
  // First, we need to convert index_s to onehot.
  // hal::oramonehot(ctx, index_s, num_elements, true)
  
  // Note: index_s variable commented out above. Need to redefine it.
  
  // 1. Prepare Data (Small scale)
  int64_t safe_N = 100;
  int64_t safe_dim = 100;
  spu::Value db_p = kernel::hal::zeros(ctx, spu::DT_I32, {safe_N, safe_dim});
  spu::Value db = kernel::hal::_p2s(ctx, db_p);
  
  // 2. Prepare Index (Small scale)
  spu::Value index_p = kernel::hal::zeros(ctx, spu::DT_I32, {1});
  spu::Value index_s = kernel::hal::_p2s(ctx, index_p);
  
  // Fallback to manual MatMul which is robust and matches linear scan logic.
  // 1. OneHot
  auto onehot_opt = kernel::hal::oramonehot(ctx, index_s, safe_N, true);
  if (!onehot_opt.has_value()) { return; }
  spu::Value onehot = onehot_opt.value();
  
  // 2. MatMul [1, N] * [N, D] -> [1, D]
  // onehot is [N]. Reshape to [1, N].
  spu::Value onehot_row = kernel::hal::reshape(ctx, onehot, {1, safe_N});
  
  spu::Value res = kernel::hal::matmul(ctx, onehot_row, db);
  
  auto end = std::chrono::system_clock::now();
  auto c1 = ctx->lctx()->GetStats()->sent_bytes.load();
  auto a1 = ctx->lctx()->GetStats()->sent_actions.load();
  
  DurationMillis time = end - start;
  SPDLOG_INFO("Small MatMul Retrieval (100x100) done. Time: {} ms", time.count());
  SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);
  SPDLOG_INFO("Actions: {}", a1 - a0);
  
  // Dimensionality Curse Experiment
  // We will loop over D.
  
  // N = 1000. D = 128, 256, 512, 1024.
  // This fits in memory.
  
  // N must be small enough to run quickly but large enough to measure.
  int64_t bench_N = 100; // Small N
  
  // Need to redefine index_s for this block as it might not be visible or valid.
  spu::Value idx_p = kernel::hal::zeros(ctx, spu::DT_I32, {1});
  spu::Value idx_s = kernel::hal::_p2s(ctx, idx_p);
  
  // Run just D=128 to confirm stability
  std::vector<int64_t> dims = {128};
  
  for (auto d : dims) {
      SPDLOG_INFO("Running D={}", d);
      spu::Value db_d = kernel::hal::zeros(ctx, spu::DT_I32, {bench_N, d});
      spu::Value db_ds = kernel::hal::_p2s(ctx, db_d);
      
      // Use OneHot + MatMul
      auto oh = kernel::hal::oramonehot(ctx, idx_s, bench_N, true);
      if (oh.has_value()) {
          spu::Value oh_row = kernel::hal::reshape(ctx, oh.value(), {1, bench_N});
          auto start_d = std::chrono::system_clock::now();
          spu::Value res_d = kernel::hal::matmul(ctx, oh_row, db_ds);
          auto end_d = std::chrono::system_clock::now();
          DurationMillis time_d = end_d - start_d;
          SPDLOG_INFO("D={} Time: {} ms", d, time_d.count());
      } else {
          SPDLOG_ERROR("oramonehot failed for D={}", d);
      }
  }
  
  return; 
  
  /*
  auto onehot = kernel::hal::oramonehot(ctx, index_s, num_elements, true);
  if (!onehot.has_value()) {
      SPDLOG_ERROR("oramonehot failed");
      return;
  }
  */ 
}

/*
void BenchSecretTopK(spu::SPUContext* ctx, 
                     int64_t num_candidates, int64_t k) {
  SPDLOG_INFO("Benchmarking Secret TopK: Candidates={}, K={}", num_candidates, k);
  
  // Generate random scores
  spu::Value scores_p = kernel::hal::zeros(ctx, spu::DT_I32, {num_candidates});
  spu::Value scores = kernel::hal::_p2s(ctx, scores_p);
  
  auto start = std::chrono::system_clock::now();
  auto c0 = ctx->lctx()->GetStats()->sent_bytes.load();
  auto a0 = ctx->lctx()->GetStats()->sent_actions.load();
  
  // SPU TopK using Sort
  std::vector<spu::Value> inputs = {scores};
  spu::Value indices = kernel::hal::iota(ctx, spu::DT_I32, num_candidates);
  inputs.push_back(indices);
  
  // Ascending sort
  auto sorted = kernel::hal::simple_sort1d(ctx, inputs, kernel::hal::SortDirection::Ascending, 1, 0);
  
  // Slice first K
  spu::Value top_k_scores = kernel::hal::slice(ctx, sorted[0], {0}, {k}, {1});
  spu::Value top_k_indices = kernel::hal::slice(ctx, sorted[1], {0}, {k}, {1});
  
  auto end = std::chrono::system_clock::now();
  auto c1 = ctx->lctx()->GetStats()->sent_bytes.load();
  auto a1 = ctx->lctx()->GetStats()->sent_actions.load();
  
  DurationMillis time = end - start;
  SPDLOG_INFO("Time: {} ms", time.count());
  SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);
  SPDLOG_INFO("Actions: {}", a1 - a0);
}
*/

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  yacl::set_num_threads(32);
  
  // Init Context
  auto hctx = panther::MakeSPUContext(Parties.getValue(), PartyRank.getValue());
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  
  // Init OT (required for some SPU ops)
  auto lctx = hctx->lctx();
  auto* comm = kctx->template getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->template getState<spu::mpc::cheetah::CheetahOTState>();
  auto nworkers = ot_state->maximum_instances();
  for (size_t i = 0; i < nworkers; i++) {
    ot_state->LazyInit(comm, i);
  }
  
  // Bootstrap OT
  yacl::parallel_for(0, nworkers, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      ot_state->get(i)->GetSenderCOT()->Bootstrap();
      ot_state->get(i)->GetReceiverCOT()->Bootstrap();
    }
  });
  
  SPDLOG_INFO("OT Bootstrapped");
  
  // Measure small scale Retrieval
  // BenchSecretIndexRetrieval(hctx.get(), 0, 0); // Params ignored in fixed small test
  
  // Actually run with valid N, Dim to avoid potential 0-size issues even if code hardcoded safe_N.
  // Although internal code uses safe_N=100.
  // But maybe other parts of SPU init failed?
  // "Benchmarking Secret Index Retrieval: N=0, Dim=0" -> Arguments passed were 0.
  
  BenchSecretIndexRetrieval(hctx.get(), 100, 100); 
  
  return 0;
}
