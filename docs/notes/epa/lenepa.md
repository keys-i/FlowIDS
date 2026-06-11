# LeNEPA

[Chemeris et al., 2026 — No-Augmentation Next-Latent Prediction](https://arxiv.org/html/2607.00958v1)

LeNEPA predicts the next time-series patch vector with a causal Transformer.
Prediction and target pass through a small projector. Gradients reach both
sides; temporal SIGReg keeps vectors from becoming identical. The projector
is removed for evaluation.

Two details matter for us. Intermediate encoder layers can give better
features than the last layer. Also, much of the study retrains one recipe on
different datasets; that is not transfer of one checkpoint. Its separate UCR
transfer check uses one pretraining seed.

It offers a next-vector comparison without an EMA teacher. But its patches
assume regular sampling, while our flows arrive irregularly and consecutive
flows may be unrelated. SIGReg also needs tuning.

Fix the evaluated layer before final testing. Do not copy the paper's claim
that vanilla NEPA needs an EMA teacher: the [original NEPA method](nepa.md)
says otherwise.

[Code](https://github.com/langotime/lenepa-milets-2026)
