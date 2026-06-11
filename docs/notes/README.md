# Paper notes

Start with the [model comparison](../plan/Model.md#experiment-sequence) to see
why each paper matters. Then use these folders:

| Folder | What to read it for |
|---|---|
| [traffic/](traffic/) | FlowTransformer, MMAE, and CMES: the closest traffic models |
| [epa/](epa/README.md) | Teacher vectors, masking, future prediction, and collapse |
| [background/](background/) | Earlier reading, datasets, standards, and software |

Reading status below was recorded on 11 September 2026. These are source
checks, not paper reproductions or measured gains for our model.

## Traffic models

| Paper | Read |
|---|---|
| [MMAE](traffic/mmae.md) | Preprocessing and method |
| [CMES](traffic/cmes.md) | Method overview in the article |
| [FlowTransformer](traffic/flowtransformer.md) | Abstract |

## EPA and JEPA

Read [the differences](epa/README.md) first: predicting hidden parts of the
current input and predicting the future are separate tasks.

| Paper | Read |
|---|---|
| [data2vec](epa/data2vec.md) | Abstract |
| [I-JEPA](epa/ijepa.md) | Abstract and EMA target method |
| [V-JEPA](epa/vjepa.md) | Method and masking comparison |
| [TS-JEPA](epa/tsjepa.md) | Method and evaluation |
| [LeJEPA](epa/lejepa.md) | Abstract and loss description |
| [NEPA](epa/nepa.md) | Method, ablations, and limitations |
| [LeNEPA](epa/lenepa.md) | Method, evaluation, and limitations |
| [HEPA](epa/hepa.md) | Method and predictor ablations |
| [NextLat](epa/nextlat.md) | Abstract |
| [Causal-JEPA](epa/causaljepa.md) | Abstract |

## Background

- [Earlier papers](background/papers.md): the August reading list, with its original limits; not rechecked here
- [Other sources](background/sources.md): standards, dataset pages, software, and the local reference repository

[Refs](../plan/Refs.md) keeps the full reference index. Add new paper notes
to the relevant folder and link them there.
