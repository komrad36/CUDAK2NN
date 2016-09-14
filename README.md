Fastest CPU implementation of both a brute-force
and a custom Multi-Index Hash Table accelerator
system for matching 512-bit binary descriptors
in 2NN mode, i.e., a match is returned if the best
match between a query vector and a training vector
is more than a certain threshold number of bits
better than the second-best match.

Yes, that means the DIFFERENCE in popcounts is used
for thresholding, NOT the ratio. This is the CORRECT
approach for binary descriptors.

Both 8-bit and 16-bit MIH tables are supported.
I currently recommend 16-bit.

All functionality is contained in the files K2NN.h and twiddle_table.h.
'main.cpp' is simply a sample test harness with example usage and
performance testing.

Example initialization of Matcher class
Matcher<false> m(tvecs, size, qvecs, size, threshold, max_twiddles);

Options:

Brute-force complete (exact) match:
m.bruteMatch();

Single twiddle pass for a very fast partial match,
with no false positives (i.e. if a match is returned, it's truly the best match):
m.fastApproxMatch();

Multi-index hash (MIH) complete (exact) match, with fall-back to brute force after max_twiddles passes:
m.exactMatch();

Match until complete or until 'n' passes elapse (partial):
m.approxMatchToNTwiddles(n);

Afterward, the finalized matches are waiting
in the vector 'm.matches'.
