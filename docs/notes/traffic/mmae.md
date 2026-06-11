# MMAE

[Liu et al., 2026 — Mean Masked Autoencoder with Flow-Mixing](https://arxiv.org/html/2603.29537v1)

MMAE works on packet bytes. It takes the first five packets of a session,
anonymises addresses and ports, and builds a fixed-length input.

The student gets a damaged version of that input. FlowMix inserts tokens
from other flows. A masking predictor uses packet statistics to select
informative regions. Training has three parts:

- reconstruct the bytes;
- identify mixed tokens;
- match the clean-flow representation from an EMA teacher.

The teacher is a copy whose weights change slowly as the student learns.
After pretraining, the student encoder is used for classification.

This is close to our M2 hybrid idea: learn exact values and a teacher's vector
together. It gives us a more useful comparison than scratch alone.

The difficulty is the input. We have flow counts and timings, not packet
bytes. Mixing whole records may break their relationships or chronology.
MMAE-NF therefore needs an explicit adaptation. The experiment is whether
our hybrid beats that adaptation and each of our two losses separately.

[Code](https://github.com/lx6c78/MMAE)
