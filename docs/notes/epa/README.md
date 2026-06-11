# EPA / JEPA differences

M1 teacher predicts a vector for the current flow from a masked input. M2 future-hybrid predicts
vectors for later flows. The papers below help separate those two tasks.

A **latent** or **embedding** is a learned vector. **JEPA** means
Joint-Embedding Predictive Architecture: predict another representation
instead of rebuilding the raw input. **Collapse** means different inputs get
the same vector, making prediction easy but the features useless.

An **EMA teacher** is a slowly updated copy of the encoder. **SIGReg** keeps
vectors spread out to prevent collapse.

| Paper | Target | Main difference for us |
|---|---|---|
| [I-JEPA](ijepa.md) | Hidden image regions | Same-input prediction with an EMA target |
| [V-JEPA](vjepa.md) | Hidden video regions | Video masking need not be future-only |
| [TS-JEPA](tsjepa.md) | Masked time-series patches | EMA targets, but sampled patches differ from flows |
| [LeJEPA](lejepa.md) | Vectors from different views | SIGReg prevents collapse without an EMA teacher |
| [NEPA](nepa.md) | Next image-patch embedding | Causal prediction with stop-gradient; weak frozen probing |
| [LeNEPA](lenepa.md) | Next time-series patch vector | Temporal SIGReg replaces stop-gradient; readout layer matters |
| [HEPA](hepa.md) | A future interval at a chosen horizon | Jointly trained targets; predictor retained and fine-tuned |
| [Causal-JEPA](causaljepa.md) | Hidden object states | Interaction modelling, not our chronological rule |
| [NextLat](nextlat.md) | Next state given the next token | The next token is supplied; our future flow cannot be |

[data2vec](data2vec.md) gives the same-input teacher-target idea behind M1 teacher.
[MMAE](../traffic/mmae.md) combines a teacher target with raw reconstruction on traffic.

Our planned M2 future-hybrid uses an EMA teacher, targets individual endpoint-related
flows, and removes the predictor after training. That differs from the
methods above. Its test is whether choosing future flows by endpoint helps
more than a matched generic future target.

`M2 future-jepa` isolates future teacher-vector prediction. `M2 future-hybrid`
uses that same objective alongside reconstruction and same-flow teacher loss.
Both use masked student history and the same later endpoint-related targets.
