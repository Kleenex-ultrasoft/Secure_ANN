#include "emp-sh2pc/emp-sh2pc.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;
using namespace emp;
using namespace std::chrono;

static inline void parse_party_and_port(char** argv, int* party, int* port) {
  *party = atoi(argv[1]);
  *port  = atoi(argv[2]);
}

static uint32_t mask_bits(int bits) {
  if (bits >= 32) return 0xFFFFFFFFu;
  return (1u << bits) - 1u;
}

int main(int argc, char** argv) {
  int port, party;
  parse_party_and_port(argv, &party, &port);
  cout << "------------------------------" << endl;

  // CLI:
  // visited_test <party> <port> [nv] [m] [id_bits] [iters] [seed]
  int nv = 512;     // |V|
  int m  = 128;     // |U|
  int id_bits = 20;
  int iters = 1;
  int seed = 0;

  if (argc >= 4) nv = atoi(argv[3]);
  if (argc >= 5) m  = atoi(argv[4]);
  if (argc >= 6) id_bits = atoi(argv[5]);
  if (argc >= 7) iters = atoi(argv[6]);
  if (argc >= 8) seed = atoi(argv[7]);

  cout << "Visited scan: nv=" << nv << " m=" << m
       << " id_bits=" << id_bits << " iters=" << iters
       << " seed=" << seed << " party=" << party << endl;

  // IMPORTANT: party-dependent seed so shares are independent
  srand(seed + 1315423911u * (unsigned)party);

  NetIO* io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port);
  setup_semi_honest(io, party);

  long total_ms = 0;
  size_t total_comm = 0;

  uint32_t mask = mask_bits(id_bits);

  // Warmup not included here (keep it simple). Add iters/warmup in runner if needed.
  for (int t = 0; t < iters; ++t) {
    // Each party generates its own shares locally
    vector<uint32_t> v_share(nv), u_share(m);
    for (int i = 0; i < nv; i++) v_share[i] = ((uint32_t)rand()) & mask;
    for (int i = 0; i < m;  i++) u_share[i] = ((uint32_t)rand()) & mask;

    vector<Integer> V; V.reserve(nv);
    vector<Integer> U; U.reserve(m);

    // Secret IDs: reconstruct in-circuit via addition of shares
    for (int j = 0; j < nv; ++j) {
      uint32_t a = (party == ALICE) ? v_share[j] : 0;
      uint32_t b = (party == BOB)   ? v_share[j] : 0;
      Integer A(id_bits, a, ALICE);
      Integer B(id_bits, b, BOB);
      V.emplace_back(A + B);
    }

    for (int i = 0; i < m; ++i) {
      uint32_t a = (party == ALICE) ? u_share[i] : 0;
      uint32_t b = (party == BOB)   ? u_share[i] : 0;
      Integer A(id_bits, a, ALICE);
      Integer B(id_bits, b, BOB);
      U.emplace_back(A + B);
    }

    // Ensure previous buffered traffic doesn't leak into measurement window
    io->flush();

    size_t c0 = io->counter;
    auto t0 = high_resolution_clock::now();

    Bit any_found(false, PUBLIC);
    for (int i = 0; i < m; ++i) {
      Bit found(false, PUBLIC);
      for (int j = 0; j < nv; ++j) {
        found = found | (U[i] == V[j]);
      }
      any_found = any_found | found;
    }

    // Tiny reveal to force full evaluation
    bool af = any_found.reveal<bool>(PUBLIC);
    (void)af;

    io->flush();

    auto t1 = high_resolution_clock::now();
    size_t c1 = io->counter;

    total_ms += (long)(duration_cast<microseconds>(t1 - t0).count() / 1000);
    total_comm += (c1 - c0);
  }

  long avg_ms = total_ms / iters;
  size_t comm_per_iter = total_comm / (size_t)iters;

  finalize_semi_honest();
  delete io;

  cout << "Visited scan time (party=" << party << "): " << avg_ms << " ms" << endl;
  cout << "Visited scan comm (party=" << party << "): " << (comm_per_iter / 1024) << " KBs" << endl;

  // Print JSON on BOTH parties; runner will sum comm across parties like your topk runner.
  cout << "{"
       << "\"framework\":\"PANTHER\","
       << "\"op\":\"visited_scan\","
       << "\"proto\":\"emp_sh2pc_gc\","
       << "\"party\":" << party << ","
       << "\"nv\":" << nv << ","
       << "\"m\":" << m << ","
       << "\"id_bits\":" << id_bits << ","
       << "\"iters\":" << iters << ","
       << "\"seed\":" << seed << ","
       << "\"lat_ms\":" << avg_ms << ","
       << "\"comm_total_bytes_reported\":" << comm_per_iter
       << "}" << endl;

  return 0;
}
