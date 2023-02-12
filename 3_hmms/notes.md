# General Notes

1. HMM Evidence Likelihood
2. Forward algorithm
3. Backward algorithm
4. Viterbi algorithm
5. Parameter learning

## HMM Evidence Likliehood
* The website uses this notation $L(X \vert \theta)$
* By the time we get here $P(HHH \vert z_{3}=F) = \Big[P(HH \vert z_{2}=F)*A_{FF}+P(HH \vert z_{2}=B)*A_{BF}\Big] \phi(H \vert F)$  we only care about two things.
  1. The probabililty of the sequences that got us the last two heads
  2. The emission probabilityat that time
* The thing is to calculate the probability of how we got those two heads we recurse down into that probability