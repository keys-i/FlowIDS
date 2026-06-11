# data2vec

[Baevski et al., 2022 — data2vec](https://proceedings.mlr.press/v162/baevski22a.html)

The student sees a masked input and predicts a vector made from the full
input. It learns to match a representation rather than reconstruct every
original value.

This is the distinction behind M1 teacher. Reconstructing an exact byte count may
be less useful than learning the surrounding traffic pattern. The risk goes
the other way too: the teacher's vector might discard a small detail that
matters for an attack.

The paper tests speech, vision, and language. For NetFlow, compare M1 teacher with
raw reconstruction and scratch. Our teacher settings are in
[Model](../../plan/Model.md#training-objectives-and-controls). Both views concern
the same event; future-event targets belong to M2 future-hybrid.

Local M1 teacher code is in `src/m1/network.py`. It uses a causal student and teacher,
Smooth L1 prediction of the top-four normalized teacher states, and a cosine
EMA schedule from 0.99 to 0.9999. CPU checks cover frozen teacher gradients,
EMA updates, and collapse rejection. This verifies our implementation choices;
it is not a reproduction of the paper's experiments.
