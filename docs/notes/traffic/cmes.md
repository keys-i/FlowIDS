# CMES cross-flow model

[Huang et al., 2026 — LLM-Driven Cross-Flow Modeling for Network Attack Traffic Detection](https://www.techscience.com/CMES/v148n1/68212/html)

CMES is the journal name. The paper groups flows before feeding them to a
model, then tells attention which flows are related. Relations include shared
endpoints, ports, protocols, and similar traffic measurements.

An MLP encodes each flow and a Transformer encodes its context. Their features
feed a frozen language-model backbone. The authors remove its causal mask,
add a learned relation term to attention, and train the added layers and
classifier.

The useful part for us is the separation between **choosing the history** and
**describing relationships inside it**. Those are our M3 and M4 comparisons.
They are already published ideas.

Our version restricts the history to completed past flows and uses endpoint
equality rather than addresses, ports, or feature similarity as relation
inputs. We need to compare those choices at the same size and training
budget. Reusing the paper's headline score would tell us little.

The earlier notes mention four-bit/16-type relations. That exact encoding
still needs checking against the method before implementing its comparison.

[Data](https://github.com/liliMpro/source_dataset)
