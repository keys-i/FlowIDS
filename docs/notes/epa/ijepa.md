# I-JEPA

[Assran et al., 2023 — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/html/2301.08243v3)

A visible image region is encoded, then a predictor estimates vectors for
hidden regions of the same image. An EMA encoder supplies the target vectors.
The paper finds that the size and placement of the masks matter.

For M1 teacher, the useful question is what to hide. Groups of related NetFlow
fields are more plausible candidates than arbitrary image-style patches,
but their usefulness still needs testing.

I-JEPA does not predict a later event. Nor does its image result tell us how
to mask irregular flows. It supplies a training idea, not a NetFlow recipe.

[Code](https://github.com/facebookresearch/ijepa)

Our `M2 future-jepa` variant uses a masked past-flow context to predict an EMA
teacher's vectors for individual later endpoint-related flows. It has no raw
reconstruction or same-flow loss. This applies the teacher/predictor idea to
future NetFlow targets; it does not reproduce I-JEPA's image blocks.
