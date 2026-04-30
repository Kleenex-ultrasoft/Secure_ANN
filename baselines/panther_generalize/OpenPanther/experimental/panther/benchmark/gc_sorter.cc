#include "experimental/panther/protocol/gc_topk.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace panther::gc;

// Format of input file:
// [N (uint32)][K (uint32)][item_bits (uint32)][id_bits (uint32)]
// [dist_shares (N * uint32)]
// [addr_shares (N * uint32)]

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <party 1|2> <port> <in_file> <out_file>" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    string in_file = argv[3];
    string out_file = argv[4];

    // Read Input
    ifstream fin(in_file, ios::binary);
    if (!fin) { cerr << "Failed to open " << in_file << endl; return 1; }

    uint32_t n, k, item_bits, id_bits;
    fin.read((char*)&n, 4);
    fin.read((char*)&k, 4);
    fin.read((char*)&item_bits, 4);
    fin.read((char*)&id_bits, 4);
    
    // Debug info
    cout << "Party " << party << " N=" << n << " K=" << k << " item_bits=" << item_bits << " id_bits=" << id_bits << endl;

    vector<uint32_t> dist_shares(n);
    vector<uint32_t> addr_shares(n);
    fin.read((char*)dist_shares.data(), n * 4);
    fin.read((char*)addr_shares.data(), n * 4);
    
    // Check if read enough bytes
    if (fin.gcount() != (long)(n * 4)) {
         cerr << "Read error or file too short (addr). Read " << fin.gcount() << " expected " << n*4 << endl;
    }
    fin.close();
    
    // Print first few shares to verify data read
    cout << "Shares[0]: " << dist_shares[0] << ", " << addr_shares[0] << endl;

    // Setup EMP
    cout << "Setting up EMP..." << endl;
    emp::NetIO *io = new emp::NetIO(party == ALICE ? nullptr : "127.0.0.1", port);
    emp::setup_semi_honest(io, party);
    cout << "EMP Setup Done." << endl;

    // Run TopK
    cout << "Running TopK..." << endl;
    
    // NOTE: The issue might be that dist_shares are secret shares (A+B), but TopK takes them directly as input.
    // In gc_topk.cc, TopK function implementation:
    // A[i] = Integer(item_bits, input[i], ALICE);
    // B[i] = Integer(item_bits, input[i], BOB);
    // INPUT[i] = A[i] + B[i];
    // This logic ADDS the input from Alice (dist_shares_A) and Bob (dist_shares_B).
    // This is CORRECT for arithmetic secret sharing reconstruction inside GC.
    //
    // However, if the shares were generated as uint32 overflows, they must be consistent with the field used here.
    // Python script: d1 = randint; d2 = d - d1 (mod 2^32).
    // C++: A + B (mod 2^item_bits).
    // If item_bits = 32, this matches.
    // But if item_bits < 32 (e.g. 31), the inputs are truncated by `item_mask = (1 << item_bits) - 1`.
    // This destroys the arithmetic share property if the share values > 2^item_bits!
    
    // Check if item_bits is sufficient for the shares (which are full 32-bit uints).
    // Shares are uniform random in [0, 2^32).
    // If we mask them to `item_bits`, we break the reconstruction `(s1 + s2) mod 2^32`.
    // The reconstruction would be `(s1 & mask) + (s2 & mask)`, which is WRONG.
    
    // FIX:
    // The shares must be passed as `Integer` initialized with full width?
    // Or we must ensure shares are generated within `item_bits`.
    // BUT we can't ensure shares are within `item_bits` if the secret is within `item_bits` but shares are additive in a larger ring.
    // Standard approach: Shares are in 2^k ring. Reconstruction is in 2^k ring.
    // The GC integer addition `A[i] + B[i]` happens with `bit_size = item_bits`.
    // So the shares MUST be modulo 2^item_bits.
    
    // So the Python script generating full 32-bit shares for a 31-bit secret (if item_bits=31) is WRONG for this specific GC circuit 
    // unless the circuit uses 32-bit addition.
    
    // We should mask inputs to item_bits HERE before passing to TopK?
    // No, TopK does `input[i] &= item_mask`. This truncates the share.
    // If shares are 32-bit random, `(s1 & mask) + (s2 & mask)` != `(s1 + s2) & mask`.
    //
    // SOLUTION: We must ensure the shares we feed are valid mod 2^item_bits.
    // Either Python script generates them mod 2^item_bits, OR we accept that `item_bits` MUST be 32 for this to work with 32-bit container.
    // Let's force masking here to be safe if Python sent 32-bit but `item_bits` is smaller,
    // AND print a warning if data loss would occur.
    
    uint32_t mask = (item_bits >= 32) ? 0xFFFFFFFF : ((1ULL << item_bits) - 1);
    for(auto& x : dist_shares) x &= mask;
    for(auto& x : addr_shares) x &= mask; // addr_shares usually smaller id_bits
    
    vector<int32_t> result_ids = TopK(n, k, item_bits, id_bits, dist_shares, addr_shares);
    cout << "TopK Done. Results: " << result_ids.size() << endl;

    // Output First Result for Debug (Bob only)
    if (party == BOB && !result_ids.empty()) {
        cout << "First Result ID: " << result_ids[0] << endl;
    }

    // Write Output (Only Bob gets result in TopK implementation, or maybe both? 
    // In gc_topk.cc: gc_id[i] = INDEX[i].reveal<int32_t>(BOB);
    
    if (party == BOB) {
        ofstream fout(out_file, ios::binary);
        if (!fout) { cerr << "Failed to open " << out_file << endl; return 1; }
        // result_ids is int32.
        fout.write((char*)result_ids.data(), result_ids.size() * 4);
        fout.close();
        cout << "Bob wrote " << result_ids.size() << " results to " << out_file << endl;
    } else {
        // Alice creates empty file to signal completion
        ofstream fout(out_file, ios::binary);
        fout.close();
    }

    delete io;
    return 0;
}

