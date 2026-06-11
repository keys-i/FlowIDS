# HEPA

[Petersen et al., 2026, v4 — Horizon-Conditioned Event Predictive Architecture](https://arxiv.org/html/2605.11130v4)

A causal encoder summarises the past. Given that vector and a time horizon,
a predictor estimates the representation of a future interval. The same
encoder reads that interval with bidirectional attention and pooling to make
the target. SIGReg prevents collapse during joint training; this version
does not use an EMA target.

For labelled event prediction, HEPA freezes the encoder and fine-tunes the
predictor. Its ablations suggest the encoder and fine-tuning setup matter
more than the predictor's initial weights.

M2 future-hybrid differs in three ways: individual endpoint-related flow targets, an EMA
teacher with causal target prefixes, and a predictor removed after training.
It is not a HEPA reproduction.

HEPA gives us the necessary comparison: predict generic future traffic with
the same anchors and training work, then check whether endpoint selection
adds anything. Its event-forecasting head is outside our current task.

[Code](https://github.com/Forgis-Labs/hepa)
