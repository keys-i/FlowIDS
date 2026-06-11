# FlowTransformer

[Manocchio et al., 2024 — FlowTransformer](https://arxiv.org/abs/2304.14746)

FlowTransformer compares ways to encode flow records, process their sequence,
and classify the result using attack labels. The paper reports that the
classification head has a large effect on performance.

That makes it a sensible M0 starting point. Base and Matched should keep the
encoding and head fixed so their comparison isolates the attention mask.
Our implementation is a PyTorch adaptation, not the original framework.

It does not answer the pretraining question. We still need the comparison
with classical models before spending time on the rest of the ladder.

[Code](https://github.com/liamdm/FlowTransformer)
