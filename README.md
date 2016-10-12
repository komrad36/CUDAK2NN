Fastest GPU implementation of a brute-force
matcher for 512-bit binary descriptors
in 2NN mode, i.e., a match is returned if the best
match between a query vector and a training vector
is more than a certain threshold number of bits
better than the second-best match.

Yes, that means the DIFFERENCE in popcounts is used
for thresholding, NOT the ratio. This is the CORRECT
approach for binary descriptors.

This laboriously crafted kernel is EXTREMELY fast.
63 BILLION comparisons per second on a stock GTX1080,
enough to match nearly 46,000 descriptors per frame at 30 fps (!)

A key insight responsible for much of the performance of
this insanely fast CUDA kernel is due to
Christopher Parker (https://github.com/csp256), to whom
I am extremely grateful.

CUDA CC 3.0 or higher is required.

All functionality is contained in the files CUDAK2NN.h
and CUDAK2NN.cu. 'main.cpp' is simply a sample test harness
with example usage and performance testing.
