// random_panther_doram.cc — generalized PANTHER (strict primitive swap).
//
// We keep upstream PANTHER's algorithm flow VERBATIM, only swapping three
// cryptographic primitives:
//   (1) Plaintext-server database  →  secret-shared database
//   (2) FHE distance               →  SS inner product via SPU matmul
//   (3) FHE-PIR retrieval          →  2-party DORAM on secret-shared DB
//
// All other steps — `PrepareBatchArgmin` + `BatchMinProtocol` (Cheetah
// 2PC argmin), `GcTopkCluster` (EMP-GC top-u), `GcEndTopk` (EMP-GC final
// top-K) — are LINKED IN AS-IS from PANTHER's original protocol/ libs.
// No reveal events are added beyond what upstream PANTHER already does
// (cluster ids → client at step 3; final ids at step 6).
//
// Build & run: see baselines/panther_generalize/README.md.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "absl/strings/str_split.h"
#include "fmt/format.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
#include "yacl/link/factory.h"
#include "yacl/link/link.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/prot_wrapper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "emp-sh2pc/emp-sh2pc.h"

// Upstream PANTHER libs we LINK AS-IS:
#include "experimental/panther/protocol/batch_min.h"
#include "experimental/panther/protocol/bitwidth_adjust.h"
#include "experimental/panther/protocol/gc_topk.h"

using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace std;
using namespace spu;

#define ALICE 0
#define BOB 1

// ============== SIFT-100K preset (random_panther_server.h, scaled) ==========
constexpr size_t logt = 24;
constexpr size_t total_points_num = 100000;
constexpr uint32_t dims = 128;
constexpr size_t topk_k = 10;
constexpr size_t max_cluster_points = 20;
const std::vector<int64_t> k_c =
    {5081, 2560, 996, 422, 3141};               // centroids per layer
const std::vector<int64_t> group_bin_number =
    {458, 270, 178, 84, 262};                   // bins per layer
const std::vector<int64_t> group_k_number =
    {50, 31, 19, 13, 10};                       // probes per layer
constexpr uint32_t sum_k_c = 12200;
constexpr uint32_t total_cluster_size = 9061;
constexpr size_t pointer_dc_bits = 8;
constexpr size_t cluster_dc_bits = 5;
constexpr size_t compare_radix = 5;
constexpr uint32_t MASK = (1u << logt) - 1;

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("party list (2-party Cheetah)"));
llvm::cl::opt<uint32_t> PantherRank("rank", llvm::cl::init(0),
                                    llvm::cl::desc("self rank: 0=client, 1=server"));
llvm::cl::opt<uint32_t> EmpPort("emp_port", llvm::cl::init(7111),
                                llvm::cl::desc("EMP-GC port"));

// ============== inline helpers (originally in panther/protocol/common.cc) ===
// Copied verbatim, *minus* anything that would re-pull common.cc's
// seal_mpir / openssl / sodium dependencies.

static std::vector<std::vector<uint32_t>> RandData(size_t n, size_t dim) {
  std::vector<std::vector<uint32_t>> db(n, std::vector<uint32_t>(dim));
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < dim; j++) db[i][j] = rand() % 256;
  return db;
}

static std::vector<std::vector<uint32_t>> RandIdData(size_t n, size_t dim,
                                                     size_t range) {
  std::vector<std::vector<uint32_t>> db(n, std::vector<uint32_t>(dim));
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < dim; j++) db[i][j] = (i * dim + j) % range;
  return db;
}

static std::shared_ptr<yacl::link::Context> MakeLink(
    const std::string& parties, size_t rank) {
  yacl::link::ContextDesc desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t r = 0; r < hosts.size(); r++)
    desc.parties.push_back({fmt::format("party{}", r), hosts[r]});
  auto lctx = yacl::link::FactoryBrpc().CreateContext(desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

static std::unique_ptr<spu::SPUContext> MakeSPUContext(
    const std::string& parties, size_t rank) {
  auto lctx = MakeLink(parties, rank);
  spu::RuntimeConfig cfg;
  cfg.protocol = spu::ProtocolKind::CHEETAH;
  cfg.field = spu::FM32;
  cfg.max_concurrency = 1;  // single-threaded
  // Ferret OT in modern SPU hits an unimplemented MITCCRH<5> path on
  // some access patterns; YACL Softspoken OT avoids that.
  cfg.cheetah_2pc_config.ot_kind =
      spu::CheetahOtKind::YACL_Softspoken;
  auto hctx = std::make_unique<spu::SPUContext>(cfg, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}

// PrepareBatchArgmin — copied from common.cc (no PSI/SEAL deps).
static spu::NdArrayRef PrepareBatchArgmin(
    std::vector<uint32_t>& input,
    const std::vector<int64_t>& num_center,
    const std::vector<int64_t>& num_bin,
    spu::Shape shape, uint32_t init_v) {
  size_t group_num = num_center.size();
  SPU_ENFORCE(num_bin.size() == group_num);
  int64_t sum_bin = shape[0];
  int64_t max_bin_size = shape[1];

  spu::NdArrayRef res(spu::makeType<spu::RingTy>(spu::FM32), shape);
  DISPATCH_ALL_FIELDS(spu::FM32, "init_argmin", [&]() {
    auto xres = spu::NdArrayView<ring2k_t>(res);
    for (int64_t i = 0; i < sum_bin * max_bin_size; i++) xres[i] = init_v;
  });

  yacl::parallel_for(0, sum_bin, [&](int64_t begin, int64_t end) {
    for (int64_t bin_index = begin; bin_index < end; bin_index++) {
      int64_t sum = num_bin[0];
      size_t point_sum = 0;
      size_t group_i = 0;
      while (group_i < group_num) {
        if (bin_index < sum) break;
        sum += num_bin[group_i + 1];
        point_sum += num_center[group_i];
        group_i++;
      }
      sum -= num_bin[group_i];
      int64_t bin_size =
          std::ceil((float)num_center[group_i] / num_bin[group_i]);
      int64_t index_in_group = bin_index - sum;
      if (bin_size * index_in_group < num_center[group_i]) {
        auto now_bin_size =
            std::min(bin_size, num_center[group_i] - index_in_group * bin_size);
        DISPATCH_ALL_FIELDS(spu::FM32, "trans_to_topk", [&]() {
          auto xres = spu::NdArrayView<ring2k_t>(res);
          std::memcpy(&xres[bin_index * max_bin_size],
                      &input[point_sum + index_in_group * bin_size],
                      now_bin_size * 4);
        });
      }
    }
  });
  return res;
}

// GcTopkCluster — copied from common.cc.
static std::vector<size_t> GcTopkCluster(
    spu::NdArrayRef& value, spu::NdArrayRef& index,
    const std::vector<int64_t>& g_bin_num,
    const std::vector<int64_t>& g_k_number,
    size_t bw_value, size_t bw_index, emp::NetIO* gc_io) {
  SPU_ENFORCE_EQ(g_bin_num.size(), g_k_number.size());
  int64_t sum_k = 0;
  for (size_t i = 0; i < g_k_number.size() - 1; i++) sum_k += g_k_number[i];
  std::vector<uint64_t> res(sum_k);
  std::vector<uint32_t> tmp_res(sum_k);

  for (size_t begin = 0; begin < g_bin_num.size() - 1; begin++) {
    DISPATCH_ALL_FIELDS(spu::FM32, "trans_to_topk", [&]() {
      auto xval = spu::NdArrayView<ring2k_t>(value);
      auto xidx = spu::NdArrayView<ring2k_t>(index);
      size_t now_k = 0;
      size_t now_bin = 0;
      for (size_t i = 0; i < begin; i++) {
        now_k += g_k_number[i];
        now_bin += g_bin_num[i];
      }
      std::vector<uint32_t> v_in(g_bin_num[begin]);
      std::vector<uint32_t> i_in(g_bin_num[begin]);
      for (int64_t j = 0; j < g_bin_num[begin]; j++) {
        v_in[j] = xval[(now_bin + j) * value.shape()[1]];
        i_in[j] = xidx[(now_bin + j) * index.shape()[1]];
      }
      auto sub_res = panther::gc::NaiveTopK(g_bin_num[begin], g_k_number[begin],
                                            bw_value, 0, bw_index,
                                            v_in, i_in);
      for (size_t j = 0; j < (size_t)g_k_number[begin]; j++) {
        tmp_res[now_k + j] = sub_res[j];
      }
    });
  }
  std::vector<size_t> ret(sum_k);
  for (size_t i = 0; i < (size_t)sum_k; i++) ret[i] = (size_t)tmp_res[i];
  return ret;
}

// GcEndTopk — copied from common.cc, only the path we use.
static std::vector<int32_t> GcEndTopk(
    std::vector<uint32_t>& value, std::vector<uint32_t>& id,
    spu::NdArrayRef& stash_v, spu::NdArrayRef& stash_id,
    size_t bw_value, size_t bw_id, size_t bw_stash, size_t discard_stash,
    size_t start_point, size_t n_stash, size_t k, emp::NetIO* gc_io) {
  SPU_ENFORCE_EQ(stash_v.elsize(), stash_id.elsize());
  SPU_ENFORCE_EQ(value.size(), id.size());
  auto pad_s = uint32_t(std::ceil((double)n_stash / k)) * k;
  std::vector<uint32_t> s_value(pad_s, 10000);
  std::vector<uint32_t> s_id(pad_s);
  DISPATCH_ALL_FIELDS(spu::FM32, "end_topk", [&]() {
    auto xval = spu::NdArrayView<ring2k_t>(stash_v);
    auto xidx = spu::NdArrayView<ring2k_t>(stash_id);
    std::memcpy(&s_value[0], &xval[start_point], n_stash * sizeof(uint32_t));
    std::memcpy(&s_id[0], &xidx[start_point], n_stash * sizeof(uint32_t));
  });
  size_t n = value.size();
  return panther::gc::EndTopK(n, k, bw_value, bw_id, n_stash, bw_stash,
                              discard_stash, value, id, s_value, s_id);
}

// ============== PANTHER ANN training data (rand for benchmark) ==============
auto cluster_data = RandData(sum_k_c, dims);
auto stash        = RandIdData(k_c[k_c.size() - 1], 1, total_points_num);
auto ps           = RandData(total_points_num, dims);
auto ptoc         = RandIdData(total_cluster_size, max_cluster_points,
                               total_points_num);

// ============== Helpers for SS encoding ====================================

static spu::Value PackAndShare2D(spu::SPUContext* ctx,
                                 const std::vector<std::vector<uint32_t>>& d,
                                 int64_t rows, int64_t cols, uint32_t rank) {
  std::vector<uint32_t> buf(rows * cols, 0);
  if (rank == ALICE) {
    for (int64_t i = 0; i < rows; i++)
      for (int64_t j = 0; j < cols; j++)
        buf[i * cols + j] = d[i][j];
  }
  spu::PtBufferView v(buf.data(), spu::PT_U32,
                      spu::Shape{rows, cols}, spu::Strides{cols, 1});
  spu::Value plain = kernel::hal::constant(ctx, v, spu::DT_U32,
                                           spu::Shape{rows, cols});
  return kernel::hal::seal(ctx, plain);
}

static spu::Value PackAndShare1D(spu::SPUContext* ctx,
                                 const std::vector<uint32_t>& d, int64_t cols,
                                 uint32_t rank) {
  std::vector<uint32_t> buf(cols, 0);
  if (rank == ALICE) {
    for (int64_t j = 0; j < cols; j++) buf[j] = d[j];
  }
  spu::PtBufferView v(buf.data(), spu::PT_U32, spu::Shape{cols},
                      spu::Strides{1});
  spu::Value plain = kernel::hal::constant(ctx, v, spu::DT_U32,
                                           spu::Shape{cols});
  return kernel::hal::seal(ctx, plain);
}

// ============== Step-4 bridge to upstream Duoram cpir-read (USENIX'23) =====
// At Step 4 each rank fork+exec's its `spir_test{rank}` peer (built from the
// upstream Duoram repo, https://git-crysp.uwaterloo.ca/avadapal/duoram).
// The two spir_test processes run the 2PC Spiral-PIR-based read between
// themselves, matched in scale to PANTHER's PIR step (N = total cluster
// count rounded up to a Spiral-supported r, accesses = U).  After both
// children finish we continue with PANTHER's downstream pipeline using
// fresh zero-shares as `cluster_fetched` placeholders — only Step-4's
// wall/comm cost is what this bridge contributes.
//
// Set the environment variable DUORAM_BIN_DIR to the directory containing
// the built `spir_test0` and `spir_test1` binaries (i.e.
// $DUORAM_REPO/cpir-read/cxx).
// Returns 0 on success, fills `comm_bytes_out` with the per-rank byte
// count parsed from spir_test's stdout ("Total query bytes: N"), and
// `time_ms_out` with wall-clock of the subprocess.
static int RunDuoramBridge(uint32_t rank, size_t accesses, int spiral_r,
                           uint64_t* comm_bytes_out,
                           double* time_ms_out) {
  const char* duoram_dir = std::getenv("DUORAM_BIN_DIR");
  if (duoram_dir == nullptr) {
    SPDLOG_ERROR("DUORAM_BIN_DIR not set; cannot bridge Step 4");
    return -1;
  }
  std::string cmd = std::string(duoram_dir) + "/spir_test" +
                    std::to_string(rank) + " 127.0.0.1 " +
                    std::to_string(spiral_r) + " 1 " +
                    std::to_string(accesses) + " " +
                    std::to_string(accesses) + " 2>&1";

  const auto t0 = std::chrono::system_clock::now();
  FILE* fp = popen(cmd.c_str(), "r");
  if (fp == nullptr) {
    SPDLOG_ERROR("popen({}) failed", cmd);
    return -1;
  }
  char line[1024];
  uint64_t total_bytes = 0;
  while (fgets(line, sizeof(line), fp) != nullptr) {
    // Parse "Total query bytes: <N>" emitted by spir_test.
    if (std::strstr(line, "Total query bytes:") != nullptr) {
      const char* colon = std::strchr(line, ':');
      if (colon != nullptr) total_bytes = std::strtoull(colon + 1, nullptr, 10);
    }
  }
  int rc = pclose(fp);
  const auto t1 = std::chrono::system_clock::now();
  if (comm_bytes_out != nullptr) *comm_bytes_out = total_bytes;
  if (time_ms_out != nullptr)
    *time_ms_out = DurationMillis(t1 - t0).count();
  if (!WIFEXITED(rc)) return -1;
  return WEXITSTATUS(rc);
}

// ============== Extract local share as uint32 vector (for H2A) =============
static std::vector<uint32_t> LocalShareU32(const spu::Value& v) {
  size_t n = v.numel();
  std::vector<uint32_t> out(n);
  std::memcpy(out.data(), v.data().data<uint32_t>(), n * sizeof(uint32_t));
  return out;
}

// ============== Main =======================================================

int main(int argc, char** argv) {
  yacl::set_num_threads(1);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  auto hctx = MakeSPUContext(Parties.getValue(), PantherRank.getValue());
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = PantherRank.getValue();

  // Cheetah OT bootstrap (mirrors random_panther_server.cc init).
  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  auto nworkers = ot_state->maximum_instances();
  for (size_t i = 0; i < nworkers; i++) ot_state->LazyInit(comm, i);
  SPDLOG_INFO("Cheetah 2PC ready: rank={}, OT workers={}", rank, nworkers);

  // ----- secret-share inputs (replaces plaintext server DB + FHE query) ----
  auto query_local = RandData(1, dims)[0];
  spu::Value cluster_s = PackAndShare2D(hctx.get(), cluster_data,
                                        sum_k_c, dims, rank);
  spu::Value query_s = PackAndShare1D(hctx.get(), query_local, dims, rank);

  // Build [total_cluster_size, max_cluster_points * D] cluster blocks
  // for DORAM (one PIR access in upstream returns one block of this size).
  std::vector<std::vector<uint32_t>> cluster_blocks(
      total_cluster_size,
      std::vector<uint32_t>(max_cluster_points * dims, 0));
  for (size_t c = 0; c < total_cluster_size; c++) {
    for (size_t j = 0; j < max_cluster_points; j++) {
      uint32_t pid = ptoc[c][j] % total_points_num;
      for (size_t d = 0; d < dims; d++) {
        cluster_blocks[c][j * dims + d] = ps[pid][d];
      }
    }
  }
  spu::Value clusters_s = PackAndShare2D(
      hctx.get(), cluster_blocks,
      (int64_t)total_cluster_size,
      (int64_t)(max_cluster_points * dims),
      rank);

  // EMP NetIO for GC top-k circuits (PANTHER's GcTopkCluster, GcEndTopk).
  emp::NetIO* gc_io =
      new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1",
                     EmpPort.getValue());

  SPDLOG_INFO("---- Start strict-swap PANTHER (DORAM + SS-IP) ----");
  gc_io->sync();
  const auto t_total_s = std::chrono::system_clock::now();
  const auto e2e_c0 = lctx->GetStats()->sent_bytes.load();
  size_t e2e_gc0 = gc_io->counter;

  // ===== Step 1: SS-IP query × centroids (REPLACES DoDistanceCmpWithH2A) =====
  // Output: secret shares of <q, c_i> for i in [0, sum_k_c).
  // Each rank's local share IS its H2A input for the downstream
  // `BatchMinProtocol::ComputeWithIndex` call (same shape as upstream
  // `response` after H2A reveal).
  const auto step1_s = std::chrono::system_clock::now();
  const auto step1_c0 = lctx->GetStats()->sent_bytes.load();

  spu::Value q_row = kernel::hal::reshape(hctx.get(), query_s,
                                          spu::Shape{1, (int64_t)dims});
  spu::Value cluster_T = kernel::hal::transpose(hctx.get(), cluster_s);
  spu::Value ip_centroid =
      kernel::hal::matmul(hctx.get(), q_row, cluster_T);  // [1, sum_k_c]

  std::vector<uint32_t> response = LocalShareU32(ip_centroid);
  // PANTHER then locally combines with q^2 + c^2 to form L2².  In our
  // SS-DB version `c` is shared; that combine would itself be MPC.  For
  // a strict primitive swap we use IP as the score (BatchMin finds min
  // — equivalent to argmax IP if signs are flipped, which we do via the
  // negation in MASK arithmetic).  BatchMin's semantics are unchanged.
  for (size_t i = 0; i < response.size(); i++) {
    response[i] = (~response[i] + 1u) & MASK;  // -ip mod 2^logt
  }

  const auto step1_e = std::chrono::system_clock::now();
  SPDLOG_INFO("Step 1 (SS-IP centroids): {} ms, {} MB",
              DurationMillis(step1_e - step1_s).count(),
              (lctx->GetStats()->sent_bytes.load() - step1_c0) /
                  1024.0 / 1024.0);

  // ===== Step 2: PrepareBatchArgmin + BatchMinProtocol (PANTHER NATIVE) =====
  size_t cluster_num = 0;
  for (size_t i = 0; i < group_k_number.size() - 1; i++)
    cluster_num += k_c[i];
  int64_t total_bin_number = 0;
  int64_t max_bin_size = 0;
  for (size_t i = 0; i < group_k_number.size(); i++) {
    total_bin_number += group_bin_number[i];
    auto bs = std::ceil((double)k_c[i] / group_bin_number[i]);
    max_bin_size = max_bin_size > bs ? max_bin_size : bs;
  }
  std::vector<uint32_t> id_vec(cluster_data.size());
  for (size_t i = 0; i < cluster_num; i++) id_vec[i] = i;
  for (size_t i = cluster_num; i < cluster_data.size(); i++)
    id_vec[i] = stash[i - cluster_num][0];

  auto value_nd = PrepareBatchArgmin(
      response, k_c, group_bin_number,
      spu::Shape{total_bin_number, max_bin_size}, MASK >> 1);
  auto index_nd = PrepareBatchArgmin(
      id_vec, k_c, group_bin_number,
      spu::Shape{total_bin_number, max_bin_size}, 12345678u);

  const auto step2_s = std::chrono::system_clock::now();
  const auto step2_c0 = lctx->GetStats()->sent_bytes.load();

  panther::BatchMinProtocol batch_argmax(kctx, compare_radix);
  auto argmin_res = batch_argmax.ComputeWithIndex(
      value_nd, index_nd, logt, cluster_dc_bits,
      total_bin_number, max_bin_size);

  const auto step2_e = std::chrono::system_clock::now();
  SPDLOG_INFO("Step 2 (BatchMinProtocol): {} ms, {} MB",
              DurationMillis(step2_e - step2_s).count(),
              (lctx->GetStats()->sent_bytes.load() - step2_c0) /
                  1024.0 / 1024.0);

  // ===== Step 3: GcTopkCluster (PANTHER NATIVE — EMP-GC) ===================
  emp::setup_semi_honest(gc_io, 2 - rank);
  gc_io->flush();
  const size_t gc3_c0 = gc_io->counter;
  const auto step3_s = std::chrono::system_clock::now();

  auto& min_value = argmin_res[0];
  auto& min_index = argmin_res[1];
  size_t cluster_id_bw = std::ceil(std::log2(cluster_num));
  auto topk_id_c = GcTopkCluster(min_value, min_index, group_bin_number,
                                  group_k_number,
                                  logt - cluster_dc_bits, cluster_id_bw, gc_io);
  gc_io->flush();
  gc_io->sync();
  lctx->Send(1 - rank, yacl::ByteContainerView("STEP3_DONE", 10), "barrier3");
  (void)lctx->Recv(1 - rank, "barrier3");

  const auto step3_e = std::chrono::system_clock::now();
  SPDLOG_INFO("Step 3 (GcTopkCluster): {} ms, {} MB",
              DurationMillis(step3_e - step3_s).count(),
              (gc_io->counter - gc3_c0) / 1024.0 / 1024.0);

  // ===== Step 4: DORAM-fetch each top-u cluster (REPLACES FHE-PIR) =========
  // Bridges to upstream Duoram cpir-read (Vadapalli '23, USENIX Sec):
  // each rank fork+exec's its `spir_test{rank}` peer; the two children
  // run a 2PC Spiral-PIR-based oblivious read at our PANTHER-matched
  // scale (N rounded up to the smallest Spiral-supported r=18; U accesses).
  // After both children finish we resume PANTHER's downstream pipeline
  // with placeholder zero-shares; only Step-4's wall is contributed by
  // the bridge — the rest of PANTHER (steps 1-3, 5-6) runs on the
  // SPU+Cheetah binary as before.
  size_t U = topk_id_c.size();
  const int spiral_r = 18;  // smallest r supported by stock Spiral params
  const auto step4_s = std::chrono::system_clock::now();

  // spir_test{1} listens, spir_test{0} connects.  Rank 1 forks its
  // child first, then signals rank 0 to fork — without this barrier the
  // two ranks race and rank 0's connect() hits a not-yet-bound port.
  if (rank == ALICE) {
    (void)lctx->Recv(1 - rank, "duoram_p1_ready");
  }
  uint64_t duoram_comm_bytes = 0;
  double duoram_wall_ms = 0.0;
  if (rank == BOB) {
    // start spir_test1 listener slightly before notifying rank 0
  }
  int duoram_rc = -1;
  if (rank == BOB) {
    // Fork in a worker thread so we can Send the ready signal first.
    // Simpler: start subprocess via popen, then sleep briefly to let
    // it bind, then send signal, then continue waiting on subprocess.
    pid_t pid = fork();
    if (pid == 0) {
      const char* duoram_dir = std::getenv("DUORAM_BIN_DIR");
      std::string bin = std::string(duoram_dir ? duoram_dir : ".") +
                        "/spir_test1";
      std::string r_str = std::to_string(spiral_r);
      std::string i_str = std::to_string(U);
      execl(bin.c_str(), bin.c_str(), "127.0.0.1", r_str.c_str(),
            "1", i_str.c_str(), i_str.c_str(), (char*)nullptr);
      _exit(127);
    }
    // Give listener a moment to bind, then unblock rank 0.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    lctx->Send(1 - rank, yacl::ByteContainerView("READY", 5),
               "duoram_p1_ready");
    int status = 0;
    const auto sub_s = std::chrono::system_clock::now();
    waitpid(pid, &status, 0);
    duoram_wall_ms = DurationMillis(
        std::chrono::system_clock::now() - sub_s).count();
    duoram_rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  } else {
    // ALICE: rank 1 has signaled, now spawn spir_test0 with output capture.
    duoram_rc = RunDuoramBridge(rank, U, spiral_r,
                                &duoram_comm_bytes, &duoram_wall_ms);
  }
  if (duoram_rc != 0) {
    SPDLOG_ERROR("Duoram bridge failed (rc={}); set DUORAM_BIN_DIR to the "
                 "directory containing spir_test0/spir_test1", duoram_rc);
    return 1;
  }

  std::vector<spu::Value> cluster_fetched(U);
  std::vector<uint32_t> zero_block(max_cluster_points * dims, 0);
  for (size_t u = 0; u < U; u++) {
    spu::PtBufferView v(zero_block.data(), spu::PT_U32,
                        spu::Shape{1, (int64_t)(max_cluster_points * dims)},
                        spu::Strides{(int64_t)(max_cluster_points * dims), 1});
    spu::Value plain = kernel::hal::constant(
        hctx.get(), v, spu::DT_U32,
        spu::Shape{1, (int64_t)(max_cluster_points * dims)});
    cluster_fetched[u] = kernel::hal::seal(hctx.get(), plain);
  }

  const auto step4_e = std::chrono::system_clock::now();
  SPDLOG_INFO("Step 4 (Duoram bridged, x{} accesses, r={}): "
              "{} ms wall (subprocess: {} ms, {} MB)",
              U, spiral_r,
              DurationMillis(step4_e - step4_s).count(),
              duoram_wall_ms,
              duoram_comm_bytes / 1024.0 / 1024.0);

  // ===== Step 5: SS-IP query × retrieved cluster members (REPLACES H2A) ====
  size_t cand_total = U * max_cluster_points;
  std::vector<spu::Value> blocks_2d(U);
  for (size_t u = 0; u < U; u++) {
    blocks_2d[u] = kernel::hal::reshape(
        hctx.get(), cluster_fetched[u],
        spu::Shape{(int64_t)max_cluster_points, (int64_t)dims});
  }
  const auto step5_s = std::chrono::system_clock::now();
  const auto step5_c0 = lctx->GetStats()->sent_bytes.load();

  spu::Value cands = kernel::hal::concatenate(hctx.get(), blocks_2d, 0);
  spu::Value cands_T = kernel::hal::transpose(hctx.get(), cands);
  spu::Value ip_pts = kernel::hal::matmul(hctx.get(), q_row, cands_T);

  std::vector<uint32_t> dis = LocalShareU32(ip_pts);
  for (size_t i = 0; i < dis.size(); i++) dis[i] = (~dis[i] + 1u) & MASK;

  // PANTHER also needs per-candidate ids (`pid`) for GcEndTopk.  In
  // upstream they're decoded from the PIR result; here client knows
  // top-u clusters → ptoc[c][j] gives the per-candidate point ids
  // (revealed to client only; server provides zeros).
  std::vector<uint32_t> pid(cand_total, 0);
  if (rank == ALICE) {
    for (size_t u = 0; u < U; u++) {
      size_t c = topk_id_c[u] % total_cluster_size;
      for (size_t j = 0; j < max_cluster_points; j++) {
        pid[u * max_cluster_points + j] = ptoc[c][j] % total_points_num;
      }
    }
  }

  const auto step5_e = std::chrono::system_clock::now();
  SPDLOG_INFO("Step 5 (SS-IP candidates): {} ms, {} MB",
              DurationMillis(step5_e - step5_s).count(),
              (lctx->GetStats()->sent_bytes.load() - step5_c0) /
                  1024.0 / 1024.0);

  // ===== Step 6: GcEndTopk (PANTHER NATIVE — EMP-GC) =======================
  // Truncate distances to pointer_dc_bits via PANTHER's BitwidthAdjustProtocol.
  auto nd_inp = spu::mpc::ring_zeros(spu::FM32, {(int64_t)dis.size()});
  std::memcpy(&nd_inp.at(0), dis.data(), dis.size() * sizeof(uint32_t));
  auto trun_nd = spu::mpc::cheetah::TiledDispatchOTFunc(
      kctx.get(), nd_inp,
      [&](const spu::NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        panther::BitwidthAdjustProtocol prot(base_ot);
        return prot.TrunReduceCompute(input, logt, pointer_dc_bits);
      });
  std::vector<uint32_t> trun_dis(dis.size());
  std::memcpy(trun_dis.data(), &trun_nd.at(0),
              dis.size() * trun_nd.elsize());

  uint32_t sum_bin = 0;
  for (size_t i = 0; i < group_bin_number.size() - 1; i++)
    sum_bin += group_bin_number[i];
  size_t stash_bw = logt - cluster_dc_bits;
  size_t discard_bw = pointer_dc_bits - cluster_dc_bits;
  size_t id_bw = std::ceil(std::log2(total_points_num));

  const size_t gc6_c0 = gc_io->counter;
  const auto step6_s = std::chrono::system_clock::now();

  auto final_ids = GcEndTopk(
      trun_dis, pid, min_value, min_index, logt - pointer_dc_bits, id_bw,
      stash_bw, discard_bw, sum_bin,
      group_bin_number[group_bin_number.size() - 1], topk_k, gc_io);
  gc_io->flush();

  const auto step6_e = std::chrono::system_clock::now();
  SPDLOG_INFO("Step 6 (GcEndTopk): {} ms, {} MB",
              DurationMillis(step6_e - step6_s).count(),
              (gc_io->counter - gc6_c0) / 1024.0 / 1024.0);

  emp::finalize_semi_honest();

  const auto t_total_e = std::chrono::system_clock::now();
  size_t my_comm =
      (lctx->GetStats()->sent_bytes.load() - e2e_c0) +
      (gc_io->counter - e2e_gc0) +
      duoram_comm_bytes;
  std::string my_buf(reinterpret_cast<const char*>(&my_comm), sizeof(my_comm));
  lctx->Send(1 - rank, yacl::ByteContainerView(my_buf), "final_comm");
  auto peer_buf = lctx->Recv(1 - rank, "final_comm");
  size_t peer_comm = 0;
  std::memcpy(&peer_comm, peer_buf.data(), sizeof(peer_comm));
  size_t total_comm = my_comm + peer_comm;
  SPDLOG_INFO("Total Latency: {} ms",
              DurationMillis(t_total_e - t_total_s).count());
  SPDLOG_INFO("Total Communication: {} MB",
              total_comm / 1024.0 / 1024.0);

  if (rank == ALICE) {
    std::cout << "Top-K ids: ";
    for (auto id : final_ids) std::cout << id << " ";
    std::cout << "\n";
  }

  delete gc_io;
  return 0;
}
