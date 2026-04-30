#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"

#include "absl/strings/str_split.h"
#include "yacl/link/link.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/base/buffer.h"

#include "../protocol/gc_topk.h"  // panther::gc::TopK

#include "libspu/core/context.h"
#include "libspu/core/type.h"  // RingTy
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"   // ring_mmul/ring_add
#include "libspu/mpc/cheetah/type.h"     // ring2k_t views

using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace spu;

// ---------------- CLI ----------------
llvm::cl::opt<std::string> PartiesOpt(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("party list host:port,host:port"));

llvm::cl::opt<uint32_t> RankOpt("rank", llvm::cl::init(0),
                                llvm::cl::desc("self rank (0 or 1)"));

llvm::cl::opt<uint32_t> EmpPortOpt("emp_port", llvm::cl::init(7111),
                                   llvm::cl::desc("emp port"));

llvm::cl::opt<uint32_t> NumTOpt("t", llvm::cl::init(1),
                                llvm::cl::desc("SPU max concurrency / threads"));

llvm::cl::opt<int64_t> MOpt("m", llvm::cl::init(128),
                            llvm::cl::desc("number of candidates (rows)"));

llvm::cl::opt<int64_t> DOpt("d", llvm::cl::init(768),
                            llvm::cl::desc("embedding dimension"));

llvm::cl::opt<int64_t> KOpt("k", llvm::cl::init(10),
                            llvm::cl::desc("top-k"));

llvm::cl::opt<int64_t> ItersOpt("iters", llvm::cl::init(3),
                                llvm::cl::desc("iters"));

llvm::cl::opt<int64_t> WarmupOpt("warmup", llvm::cl::init(1),
                                 llvm::cl::desc("warmup iters"));

llvm::cl::opt<int64_t> ElemBitsOpt("elem_bits", llvm::cl::init(8),
                                   llvm::cl::desc("input element bitwidth (shares masked)"));

llvm::cl::opt<int64_t> ItemBitsOpt("item_bits", llvm::cl::init(31),
                                   llvm::cl::desc("GC distance bitwidth (signed-safe)"));

llvm::cl::opt<int64_t> IdBitsOpt("id_bits", llvm::cl::init(20),
                                 llvm::cl::desc("GC id bitwidth"));

llvm::cl::opt<uint32_t> SeedOpt("seed", llvm::cl::init(0),
                                llvm::cl::desc("seed"));

// ---------- link tuning knobs ----------
llvm::cl::opt<std::string> LinkIdOpt(
    "link_id", llvm::cl::init("root"),
    llvm::cl::desc("YACL link context id (default: root)"));

llvm::cl::opt<uint64_t> LinkRecvTimeoutMsOpt(
    "link_recv_timeout_ms", llvm::cl::init(600000), 
    llvm::cl::desc("YACL link recv timeout in ms"));

llvm::cl::opt<uint32_t> LinkConnectRetryTimesOpt(
    "link_connect_retry_times", llvm::cl::init(60),
    llvm::cl::desc("YACL connect retry times"));

llvm::cl::opt<uint32_t> LinkConnectRetryIntervalMsOpt(
    "link_connect_retry_interval_ms", llvm::cl::init(1000),
    llvm::cl::desc("YACL connect retry interval ms"));

llvm::cl::opt<uint32_t> LinkThrottleWindowSizeOpt(
    "link_throttle_window_size", llvm::cl::init(0),
    llvm::cl::desc("ContextDesc.throttle_window_size override (0=keep default)"));

llvm::cl::opt<uint32_t> LinkChunkParallelSendSizeOpt(
    "link_chunk_parallel_send_size", llvm::cl::init(8),
    llvm::cl::desc("ContextDesc.chunk_parallel_send_size"));

llvm::cl::opt<bool> LinkExitIfAsyncErrorOpt(
    "link_exit_if_async_error", llvm::cl::init(true),
    llvm::cl::desc("ContextDesc.exit_if_async_error"));

// ---------------- Helpers ----------------

// [关键修复] 手动实现的 Barrier，替代 WaitLinkTaskFinish，避免锁死通道
static void LinkBarrier2P(const std::shared_ptr<yacl::link::Context>& lctx,
                          size_t rank, std::string_view tag) {
  const size_t peer = 1 - rank;
  std::string tag01 = fmt::format("barrier/{}/0->1", tag);
  std::string tag10 = fmt::format("barrier/{}/1->0", tag);
  
  // 使用静态数据防止野指针，虽然 ByteContainerView 只是视图
  static const uint8_t kOne = 1;
  yacl::ByteContainerView payload(&kOne, 1);

  if (rank == 0) {
    lctx->Send(peer, payload, tag01);
    (void)lctx->Recv(peer, tag10);
  } else {
    (void)lctx->Recv(peer, tag01);
    lctx->Send(peer, payload, tag10);
  }
}

static std::shared_ptr<yacl::link::Context> MakeLinkBrpc(const std::string& parties,
                                                         size_t rank) {
  yacl::link::ContextDesc desc;
  desc.id = LinkIdOpt.getValue();
  desc.recv_timeout_ms = LinkRecvTimeoutMsOpt.getValue();
  desc.connect_retry_times = LinkConnectRetryTimesOpt.getValue();
  desc.connect_retry_interval_ms = LinkConnectRetryIntervalMsOpt.getValue();

  if (LinkThrottleWindowSizeOpt.getValue() != 0) {
    desc.throttle_window_size = LinkThrottleWindowSizeOpt.getValue();
  }
  desc.chunk_parallel_send_size = LinkChunkParallelSendSizeOpt.getValue();
  desc.exit_if_async_error = LinkExitIfAsyncErrorOpt.getValue();

  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t r = 0; r < hosts.size(); r++) {
    desc.parties.push_back({fmt::format("party{}", r), hosts[r]});
  }

  auto lctx = yacl::link::FactoryBrpc().CreateContext(desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

static std::unique_ptr<spu::SPUContext> MakeSPUContextSoftspoken(
    const std::string& parties, size_t rank, size_t num_thread) {
  auto lctx = MakeLinkBrpc(parties, rank);

  spu::RuntimeConfig cfg;
  cfg.set_protocol(spu::ProtocolKind::CHEETAH);
  cfg.set_field(spu::FM32);
  cfg.set_max_concurrency(num_thread);
  cfg.mutable_cheetah_2pc_config()->set_ot_kind(spu::CheetahOtKind::YACL_Softspoken);

  auto hctx = std::make_unique<spu::SPUContext>(cfg, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}

static inline uint32_t mask_bits(int bits) {
  if (bits >= 32) return 0xFFFFFFFFu;
  return (1u << bits) - 1u;
}

static NdArrayRef RandShareFM32(const Shape& shape, uint32_t seed, int elem_bits) {
  NdArrayRef a(makeType<RingTy>(spu::FM32), shape);
  std::mt19937 rng(seed);
  uint32_t mask = mask_bits(elem_bits);

  DISPATCH_ALL_FIELDS(spu::FM32, "fill_rand_share", [&]() {
    auto v = NdArrayView<ring2k_t>(a);
    for (int64_t i = 0; i < a.numel(); i++) {
      v[i] = static_cast<ring2k_t>(rng() & mask);
    }
  });
  return a;
}

static NdArrayRef Transpose2D_FM32(const NdArrayRef& in) {
  SPU_ENFORCE(in.shape().size() == 2, "Transpose2D expects a 2D tensor");
  const int64_t rows = in.shape()[0];
  const int64_t cols = in.shape()[1];

  NdArrayRef out(makeType<RingTy>(spu::FM32), {cols, rows});

  DISPATCH_ALL_FIELDS(spu::FM32, "transpose2d_fm32", [&]() {
    auto vin = NdArrayView<ring2k_t>(in);
    auto vout = NdArrayView<ring2k_t>(out);
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        vout[j * rows + i] = vin[i * cols + j];
      }
    }
  });

  return out;
}

static inline int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t mid = v.size() / 2;
  if (v.size() & 1) return v[mid];
  return 0.5 * (v[mid - 1] + v[mid]);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  
  spdlog::flush_on(spdlog::level::info);

  const uint32_t rank = RankOpt.getValue();
  const int64_t m = MOpt.getValue();
  const int64_t d = DOpt.getValue();
  const int64_t k = KOpt.getValue();
  const int64_t iters = ItersOpt.getValue();
  const int64_t warmup = WarmupOpt.getValue();
  const int elem_bits = static_cast<int>(ElemBitsOpt.getValue());
  const int item_bits = static_cast<int>(ItemBitsOpt.getValue());
  const int id_bits = static_cast<int>(IdBitsOpt.getValue());
  const size_t num_thread = NumTOpt.getValue();

  SPU_ENFORCE(m > 0 && d > 0 && k > 0);
  yacl::set_num_threads(num_thread);

  const int64_t mpad = ceil_div(m, k) * k;

  // SPU Init
  auto hctx = MakeSPUContextSoftspoken(PartiesOpt.getValue(), rank, num_thread);
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();

  // SPU Precompute (Warmup / KeyGen)
  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  auto nworkers = ot_state->maximum_instances();
  for (size_t i = 0; i < nworkers; i++) ot_state->LazyInit(comm, i);

  auto* dot = kctx->getState<spu::mpc::cheetah::CheetahDotState>()->get();
  dot->LazyInitKeys(spu::FM32);

  std::vector<double> ms_total, ms_spu, ms_gc;
  uint64_t spu_bytes_sum = 0;
  uint64_t spu_actions_sum = 0;
  uint64_t gc_bytes_sum = 0;

  using spu::mpc::cheetah::Shape3D;

  auto run_one = [&](uint32_t seed_base, bool measure) {
    NdArrayRef X = RandShareFM32({m, d}, seed_base + 1315423911u * rank + 1, elem_bits);
    NdArrayRef q = RandShareFM32({d, 1}, seed_base + 2654435761u * rank + 7, elem_bits);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto s_r0 = lctx->GetStats()->sent_actions.load();
    auto s_c0 = lctx->GetStats()->sent_bytes.load();

    NdArrayRef local = spu::mpc::ring_mmul(X, q);
    
    // Cross Term 1
    NdArrayRef cross01 = (rank == 0)
        ? dot->DotOLE(X, Shape3D{m, d, 1}, /*is_lhs=*/true)
        : dot->DotOLE(q, Shape3D{m, d, 1}, /*is_lhs=*/false);

    // Cross Term 2 (Transpose trick)
    NdArrayRef cross10_col;
    if (rank == 0) {
      NdArrayRef qT = Transpose2D_FM32(q);
      NdArrayRef cross10_row = dot->DotOLE(qT, Shape3D{1, d, m}, /*is_lhs=*/true);
      cross10_col = Transpose2D_FM32(cross10_row);
    } else {
      NdArrayRef XT = Transpose2D_FM32(X);
      NdArrayRef cross10_row = dot->DotOLE(XT, Shape3D{1, d, m}, /*is_lhs=*/false);
      cross10_col = Transpose2D_FM32(cross10_row);
    }

    NdArrayRef scores = spu::mpc::ring_add(local, cross01);
    spu::mpc::ring_add_(scores, cross10_col);

    // [关键修复] 使用 Send/Recv 屏障，而不是 WaitLinkTaskFinish
    // 这样可以确保同步，但不会干扰后台的 DotOLE 异步发送
    LinkBarrier2P(lctx, rank, "spu_done_barrier");
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto s_r1 = lctx->GetStats()->sent_actions.load();
    auto s_c1 = lctx->GetStats()->sent_bytes.load();

    std::vector<uint32_t> dist_share(mpad);
    DISPATCH_ALL_FIELDS(spu::FM32, "export_scores", [&]() {
      auto v = NdArrayView<ring2k_t>(scores);
      for (int64_t i = 0; i < m; i++) dist_share[i] = static_cast<uint32_t>(v[i]);
    });
    uint32_t padv = (1u << (item_bits - 1)) - 1u;
    for (int64_t i = m; i < mpad; i++) dist_share[i] = (rank == 0) ? padv : 0u;
    std::vector<uint32_t> id_share(mpad, 0);
    for (int64_t i = 0; i < m; i++) id_share[i] = (rank == 1) ? static_cast<uint32_t>(i) : 0u;

    // EMP / GC Init
    const int emp_role = (rank == 0) ? emp::ALICE : emp::BOB;
    const char* emp_addr = (emp_role == emp::ALICE) ? "127.0.0.1" : nullptr;

    SPDLOG_INFO("[rank={}] SPU done. Connect EMP...", rank);
    
    emp::NetIO* gc_io = new emp::NetIO(emp_addr, EmpPortOpt.getValue());
    emp::setup_semi_honest(gc_io, emp_role);
    
    gc_io->flush();
    size_t g0 = gc_io->counter;
    auto g_t0 = std::chrono::high_resolution_clock::now();

    auto topk_ids = panther::gc::TopK(
        static_cast<size_t>(mpad),
        static_cast<size_t>(k),
        static_cast<size_t>(item_bits),
        static_cast<size_t>(id_bits),
        dist_share,
        id_share);

    auto g_t1 = std::chrono::high_resolution_clock::now();
    gc_io->flush();
    size_t g1 = gc_io->counter;

    emp::finalize_semi_honest();
    delete gc_io;

    double spu_dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    uint64_t spu_bytes = (s_c1 - s_c0);
    uint64_t spu_actions = (s_r1 - s_r0);
    double gc_dt = std::chrono::duration<double, std::milli>(g_t1 - g_t0).count();
    uint64_t gc_bytes = (g1 - g0);

    if (measure) {
      ms_spu.push_back(spu_dt);
      ms_gc.push_back(gc_dt);
      ms_total.push_back(spu_dt + gc_dt);
      spu_bytes_sum += spu_bytes;
      spu_actions_sum += spu_actions;
      gc_bytes_sum += gc_bytes;
    }
    
    if (rank == 0 && measure && !topk_ids.empty()) {
       (void)topk_ids[0];
    }
  };

  for (int64_t i = 0; i < warmup; i++) run_one(SeedOpt.getValue() + i, false);
  for (int64_t i = 0; i < iters; i++) run_one(SeedOpt.getValue() + warmup + i, true);

  std::cout << "{"
            << "\"framework\":\"PANTHER\","
            << "\"op\":\"mmul_topk_pipeline\","
            << "\"rank\":" << rank << ","
            << "\"proto_spu\":\"CHEETAH\","
            << "\"proto_gc\":\"emp_sh2pc_gc\","
            << "\"m\":" << m << ","
            << "\"d\":" << d << ","
            << "\"k\":" << k << ","
            << "\"iters\":" << iters << ","
            << "\"lat_total_ms_median\":" << median(ms_total) << ","
            << "\"lat_spu_ms_median\":" << median(ms_spu) << ","
            << "\"lat_gc_ms_median\":" << median(ms_gc) << ","
            << "\"spu_comm_bytes_sum\":" << spu_bytes_sum << ","
            << "\"spu_actions_sum\":" << spu_actions_sum << ","
            << "\"gc_comm_bytes_sum\":" << gc_bytes_sum
            << "}\n";
  // 1. 确认双方都跑完了循环
  LinkBarrier2P(lctx, rank, "all_done");
  // 2. 只有在确认对方完成后，才清理本地的发送队列
  lctx->WaitLinkTaskFinish();

  return 0;
}