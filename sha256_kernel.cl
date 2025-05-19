// sha256_kernel.cl

/*
 * A basic SHA-256 implementation for OpenCL.
 * This is for demonstration and benchmarking, not cryptographic security best-practices.
 * Based on FIPS 180-4.
 */

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n)  ((x) >> (n))

#define Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define Sigma0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define Sigma1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0_small(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3)) // Renamed to avoid conflict if included elsewhere
#define sigma1_small(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10)) // Renamed

// SHA-256 constants (first 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311)
// MODIFIED LINE: Added __constant
__constant static const uint k[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Function to perform SHA-256 padding and hashing on a 4-byte nonce
// The message will be the nonce itself (4 bytes)
// Output is a 32-byte hash (8 uints)
void sha256_transform_nonce(uint nonce_val, __global uint* hash_out) { // Renamed nonce to nonce_val to avoid conflict with potential future global nonce
    uint h_state[8] = { // Initial hash values (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19)
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint w[64];
    uint message_block[16]; // 512-bit block

    // Prepare the message block (only one block for a 4-byte nonce_val)
    // Nonce is the first 4 bytes
    message_block[0] = nonce_val; // Input nonce_val
    message_block[1] = 0x80000000; // Padding: append '1' bit (MSB of the next byte) followed by zeros
    // Clear remaining message block words up to the length fields
    for (int i = 2; i < 14; ++i) {
        message_block[i] = 0x00000000; // Zero padding
    }
    // Message length in bits (4 bytes = 32 bits)
    // Stored in the last 64 bits (2 uints) of the 512-bit block, big-endian
    message_block[14] = 0x00000000;
    message_block[15] = 32;         // Length = 32 bits

    // Initialize W schedule array
    for (int t = 0; t < 16; ++t) {
        // For this benchmark, we assume the nonce_val (uint) is effectively treated as big-endian by the operations.
        // A robust crypto implementation would handle endian conversion explicitly if the source bytes were chars.
        w[t] = message_block[t];
    }

    for (int t = 16; t < 64; ++t) {
        w[t] = sigma1_small(w[t-2]) + w[t-7] + sigma0_small(w[t-15]) + w[t-16];
    }

    uint a = h_state[0];
    uint b = h_state[1];
    uint c = h_state[2];
    uint d = h_state[3];
    uint e = h_state[4];
    uint f = h_state[5];
    uint g = h_state[6];
    uint H_var = h_state[7]; // Renamed to avoid conflict with h_state array

    for (int t = 0; t < 64; ++t) {
        uint T1 = H_var + Sigma1(e) + Ch(e, f, g) + k[t] + w[t];
        uint T2 = Sigma0(a) + Maj(a, b, c);
        H_var = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    h_state[0] += a;
    h_state[1] += b;
    h_state[2] += c;
    h_state[3] += d;
    h_state[4] += e;
    h_state[5] += f;
    h_state[6] += g;
    h_state[7] += H_var;

    for (int i = 0; i < 8; ++i) {
        hash_out[i] = h_state[i];
    }
}

__kernel void hash_main(__global uint* output_hashes, uint start_nonce, uint num_hashes_per_kernel_call) {
    int gid = get_global_id(0);

    if (gid < num_hashes_per_kernel_call) {
        uint current_nonce = start_nonce + gid;
        // Each hash is 8 uints (32 bytes)
        // The output for work-item 'gid' starts at index 'gid * 8'
        sha256_transform_nonce(current_nonce, &output_hashes[gid * 8]);
    }
}