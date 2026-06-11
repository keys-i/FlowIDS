# Causal-JEPA

[Nam et al., 2026 — Learning World Models through Object-Level Latent Masking](https://arxiv.org/abs/2602.11389)

Causal-JEPA hides object-level vectors and predicts each hidden object's
state from the surrounding objects. The mask forces the model to use
interactions that an easier, fully visible input might let it ignore.

For us, that is a reason to think carefully about the unit being masked.
It is not a reason to treat endpoints as the paper's object representation.

“Causal” here concerns interaction-dependent prediction and controlled
visibility. Our causal rule concerns time: no later flow can enter the
student input. The names do not make those claims equivalent.

[Code](https://github.com/galilai-group/cjepa)
