# NextLat

[Teoh et al., 2025, revised June 2026 — Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/abs/2511.05963)

NextLat adds an internal-state prediction loss to next-token training. It
predicts the next state **given the next token**, encouraging the model to
compress its history into a state with consistent updates.

That last condition separates it from M2 future-hybrid. Our student must predict a later
flow's vector without seeing that flow. Supplying the next flow would change
the experiment and violate the planned input rule.

It gives us background on state learning. Its language and world-model
results do not answer the NetFlow transfer test.

[Code](https://github.com/JaydenTeoh/NextLat)
