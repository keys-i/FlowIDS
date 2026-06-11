# NEPA

[Xu et al., 2025 — Next-Embedding Prediction Makes Strong Vision Learners](https://arxiv.org/html/2512.16922v2)

NEPA orders image patches and predicts the next patch embedding using causal
attention. Gradients are stopped at the target; no pixel decoder or momentum
target encoder is needed. Reporting an EMA-averaged model is separate from
using an EMA teacher to generate targets.

The concern for our project is its weak standard linear probing despite
strong fine-tuning results. A model can predict embeddings well without
leaving a useful final vector for a small frozen classifier.

The objective is a simple reference for next-vector learning, but image-patch
order is not network time. For NetFlow we need both frozen and fine-tuned
comparisons, and a reason to expect the selected next flow to be relevant.

NEPA stands for Next-Embedding Predictive Autoregression. It is different
from [NextLat](nextlat.md).
