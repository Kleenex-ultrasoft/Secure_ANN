#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>

#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"

using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9540,127.0.0.1:9541"),
    llvm::cl::desc("party list host:port,host:port"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

llvm::cl::opt<int64_t> M("m", llvm::cl::init(128),
                         llvm::cl::desc("number of candidates (rows)"));

llvm::cl::opt<int64_t> D("d", llvm::cl::init(128),
                         llvm::cl::desc("embedding dimension"));

llvm::cl::opt<int64_t> Iters("iters", llvm::cl::init(20),
                             llvm::cl::desc("repeat times"));

llvm::cl::opt<int64_t> ElemBits("elem_bits", llvm::cl::init(8),
                                llvm::cl::desc("quantized element bitwidth"));

llvm::cl::opt<int64_t> DiscardBits("discard_bits", llvm::cl::init(5),
                                   llvm::cl::desc("optional right shift on scores"));

llvm::cl::opt<uint32_t> NumT("t", llvm::cl::init(1),
                             llvm::cl::desc("SPU max concurrency / threads"));

llvm::cl::opt<uint32_t> Seed("seed", llvm::cl::init(0),
                             llvm::cl::desc("local seed (per-rank diversified)"));

static std::shared_ptr<yacl::link::Context> MakeLink(const std::string& parties,
                                                     size_t rank) {
  yacl::link::ContextDesc desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t r = 0; r < hosts.size(); r++) {
    desc.parties.push_back({std::string("party") + std::to_string(r), hosts[r]});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

static std::unique_ptr<spu::SPUContext> MakeSPUContext(size_t max_conc) {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());
  spu::RuntimeConfig cfg;
  cfg.set_protocol(spu::ProtocolKind::CHEETAH);
  cfg.set_field(spu::FM32);
  cfg.set_max_concurrency(max_conc);

  // ✅ Ferret crashes on your machine; Softspoken works
  cfg.mutable_cheetah_2pc_config()->set_ot_kind(spu::CheetahOtKind::YACL_Softspoken);

  auto hctx = std::make_unique<spu::SPUContext>(cfg, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}

static inline uint32_t bitmask_u32(int64_t bits) {
  if (bits >= 32) return 0xFFFFFFFFu;
  return (1u << bits) - 1u;
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  const int64_t m = M.getValue();
  const int64_t d = D.getValue();
  const int64_t iters = Iters.getValue();
  const int64_t elem_bits = ElemBits.getValue();
  const int64_t discard_bits = DiscardBits.getValue();
  const size_t max_conc = NumT.getValue();
  const auto rank = Rank.getValue();

  auto hctx = MakeSPUContext(max_conc);
  auto lctx = hctx->lctx();
  lctx->SetThrottleWindowSize(0);

  SPDLOG_INFO("SS-IP bench rank={} m={} d={} iters={} elem_bits={} discard_bits={} max_conc={} seed={}",
              rank, m, d, iters, elem_bits, discard_bits, max_conc, Seed.getValue());

  // Initialize OT state (Panther style)
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  auto nworkers = ot_state->maximum_instances();
  for (size_t i = 0; i < nworkers; i++) {
    ot_state->LazyInit(comm, i);
  }
  lctx->WaitLinkTaskFinish();

  // ===== Avoid ring_rand (can crash on your env). Fill shares manually. =====
  uint32_t emask = bitmask_u32(elem_bits);
  // diversify seeds across ranks so shares are independent
  std::srand((unsigned)(Seed.getValue() + 1315423911u * rank));

  auto X = spu::mpc::ring_zeros(spu::FM32, {m, d});
  auto q = spu::mpc::ring_zeros(spu::FM32, {d});

  // Fill local shares with small values (quantized)
  for (int64_t i = 0; i < m * d; ++i) {
    X.at<uint32_t>(i) = ((uint32_t)std::rand()) & emask;
  }
  for (int64_t j = 0; j < d; ++j) {
    q.at<uint32_t>(j) = ((uint32_t)std::rand()) & emask;
  }

  // Replicate q into [m,d] locally
  auto Q = spu::mpc::ring_zeros(spu::FM32, {m, d});
  for (int64_t i = 0; i < m; ++i) {
    std::memcpy(&Q.at(i * d), &q.at(0), d * sizeof(uint32_t));
  }

  // Warmup
  (void)spu::mpc::ring_mul(X, Q);
  lctx->WaitLinkTaskFinish();

  auto r0 = lctx->GetStats()->sent_actions.load();
  auto c0 = lctx->GetStats()->sent_bytes.load();
  const auto t0 = std::chrono::system_clock::now();

  for (int64_t it = 0; it < iters; ++it) {
    auto prod = spu::mpc::ring_mul(X, Q);  // [m,d]

    // local reduction: scores[i] = sum_j prod[i,j]
    auto scores = spu::mpc::ring_zeros(spu::FM32, {m});
    for (int64_t i = 0; i < m; ++i) {
      uint32_t acc = 0;
      const int64_t base = i * d;
      for (int64_t j = 0; j < d; ++j) {
        acc += prod.at<uint32_t>(base + j);
      }
      scores.at<uint32_t>(i) = acc;
    }

    if (discard_bits > 0) {
      auto shifted = spu::mpc::ring_rshift(scores, discard_bits);
      (void)shifted;
    }
  }

  lctx->WaitLinkTaskFinish();

  const auto t1 = std::chrono::system_clock::now();
  auto r1 = lctx->GetStats()->sent_actions.load();
  auto c1 = lctx->GetStats()->sent_bytes.load();

  const DurationMillis dt = t1 - t0;
  double ms_per_iter = dt.count() / (double)iters;
  double mb_per_iter = (c1 - c0) / 1024.0 / 1024.0 / (double)iters;
  double act_per_iter = (r1 - r0) / (double)iters;

  SPDLOG_INFO("SS-IP time: {:.3f} ms/iter", ms_per_iter);
  SPDLOG_INFO("SS-IP comm: {:.3f} MB/iter, actions: {:.2f}/iter", mb_per_iter, act_per_iter);

  if (rank == 0) {
    std::cout << "{"
              << "\"framework\":\"PANTHER\","
              << "\"op\":\"ss_ip\","
              << "\"proto\":\"spu_cheetah_fm32_softspoken\","
              << "\"m\":" << m << ","
              << "\"d\":" << d << ","
              << "\"iters\":" << iters << ","
              << "\"elem_bits\":" << elem_bits << ","
              << "\"discard_bits\":" << discard_bits << ","
              << "\"seed\":" << Seed.getValue() << ","
              << "\"lat_ms_per_iter\":" << ms_per_iter << ","
              << "\"comm_mb_per_iter\":" << mb_per_iter << ","
              << "\"actions_per_iter\":" << act_per_iter
              << "}\n";
  }

  return 0;
}
