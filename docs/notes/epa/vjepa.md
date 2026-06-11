# V-JEPA

[Bardes et al., 2024 — Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/html/2404.08471v1)

V-JEPA predicts vectors for hidden video regions from visible regions,
without reconstructing pixels. An EMA target encoder supplies the vectors.
The learned encoder is then tested on appearance and motion tasks.

The masking distinction matters here. The paper compares masks spread across
the clip with a version restricted to early frames; these are different
prediction tasks. Video input alone does not make the method future-only.

Its frozen-encoder tests are relevant to our aim: useful pretraining should
leave features that work with few labels. The video patch layout and regular
frame timing do not directly fit flow records. Use it to motivate the latent
target comparison, not as evidence that our M2 future-hybrid will work.

[Code](https://github.com/facebookresearch/jepa)
