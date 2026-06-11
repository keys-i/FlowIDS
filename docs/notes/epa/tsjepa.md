# TS-JEPA

[Ennadir et al., 2025 arXiv version — Joint Embeddings Go Temporal](https://arxiv.org/html/2509.25449v1)

TS-JEPA splits a time series into patches, encodes the visible patches, and
predicts vectors for the hidden ones. An EMA encoder supplies the targets.
It uses a 1D convolution to form patches and adds position embeddings, then
evaluates classification and forecasting with the encoder frozen.

This is closer to our measurements than image JEPA, but sampled sensor
patches are still different from irregular flow events. Uniform masking also
allows a different information pattern from our past-only context. A
forecasting test does not make the pretraining objective causal forecasting.

For M1 teacher, the useful comparison is hidden values versus hidden vectors. The
patching and masking choices need to be adapted rather than copied.

[Code](https://github.com/Sennadir/TS_JEPA)
