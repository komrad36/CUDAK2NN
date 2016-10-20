/*******************************************************************
*   CUDAK2NN.cu
*   CUDAK2NN
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 12, 2016
*******************************************************************/
//
// Fastest GPU implementation of a brute-force
// matcher for 512-bit binary descriptors
// in 2NN mode, i.e., a match is returned if the best
// match between a query vector and a training vector
// is more than a certain threshold number of bits
// better than the second-best match.
//
// Yes, that means the DIFFERENCE in popcounts is used
// for thresholding, NOT the ratio. This is the CORRECT
// approach for binary descriptors.
//
// This laboriously crafted kernel is EXTREMELY fast.
// 63 BILLION comparisons per second on a stock GTX1080,
// enough to match nearly 46,000 descriptors per frame at 30 fps (!)
//
// A key insight responsible for much of the performance of
// this insanely fast CUDA kernel is due to
// Christopher Parker (https://github.com/csp256), to whom
// I am extremely grateful.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files CUDAK2NN.h
// and CUDAK2NN.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#include "CUDAK2NN.h"

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDAK2NN_kernel(const cudaTextureObject_t tex_q, const int num_q, const uint64_t* __restrict__ g_training, const int num_t, int* const __restrict__ g_match, const int threshold) {
	uint64_t train = *(g_training += threadIdx.x & 7);
	g_training += 8;
	uint64_t q[8];
	for (int i = 0, offset = ((threadIdx.x & 24) << 3) + (threadIdx.x & 7) + (blockIdx.x << 11) + (threadIdx.y << 8); i < 8; ++i, offset += 8) {
		const uint2 buf = tex1Dfetch<uint2>(tex_q, offset);
		asm("mov.b64 %0, {%1,%2};" : "=l"(q[i]) : "r"(buf.x), "r"(buf.y)); // some assembly required
	}	
	int best_i, best_v = 100000, second_v = 200000;
#pragma unroll 6
	for (int t = 0; t < num_t; ++t, g_training += 8) {
		uint32_t dist[4];
		for (int i = 0; i < 4; ++i) dist[i] = __byte_perm(__popcll(q[i] ^ train), __popcll(q[i + 4] ^ train), 0x5410);
		for (int i = 0; i < 4; ++i) dist[i] += __shfl_xor(dist[i], 1);
		train = *g_training;
		if (threadIdx.x & 1) dist[0] = dist[1];
		if (threadIdx.x & 1) dist[2] = dist[3];
		dist[0] += __shfl_xor(dist[0], 2);
		dist[2] += __shfl_xor(dist[2], 2);
		if (threadIdx.x & 2) dist[0] = dist[2];
		dist[0] = __byte_perm(dist[0] + __shfl_xor(dist[0], 4), 0, threadIdx.x & 4 ? 0x5432 : 0x5410);
		second_v = min(dist[0], second_v);
		if (dist[0] < best_v) {
			second_v = best_v;
			best_i = t;
			best_v = dist[0];
		}
	}
	const int idx = (blockIdx.x << 8) + (threadIdx.y << 5) + threadIdx.x;
	if (idx < num_q) g_match[idx] = second_v - best_v > threshold ? best_i : -1;
}

void CUDAK2NN(const void* const __restrict d_t, const int num_t, const cudaTextureObject_t tex_q, const int num_q, int* const __restrict d_m, const int threshold) {
	CUDAK2NN_kernel<<<((num_q - 1) >> 8) + 1, { 32, 8 }>>>(tex_q, num_q, reinterpret_cast<const uint64_t*>(d_t), num_t, d_m, threshold);
	cudaDeviceSynchronize();
}
