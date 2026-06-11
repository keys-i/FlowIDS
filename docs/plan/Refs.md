# Research evidence and novelty notes

**Cutoff:** 2026-08-28. I use primary papers, official repositories, standards,
and dataset pages. These notes track novelty; I have not reproduced every
paper. `✅` locally reproduced, `🟡` at least one cited implementation or
checkpoint exists but I have not reproduced the row locally, `❌` no usable
model artifact, `❓` unresolved identity.

## Closest work and reproduction status

### Closest work

| Method | Closest work | Shared idea | Important difference | Reproduction status | What follows |
|---|---|---|---|---|---|
| M1-R | [MMAE2026](#mmae2026), [GraphIDS](#graphids) | Masked traffic reconstruction | I hide semantic NetFlow feature groups in a past-only flow context. | 🟡 | Masked-flow learning is prior art. M1-R must beat scratch without identifiers or ports. |
| M1-L | [MMAE2026](#mmae2026), [data2vec](#data2vec) | Corrupted student, unmasked EMA teacher | I predict the same completed NetFlow event state from causal context. | 🟡 | EMA latent targets are prior art. Collapse or weak transfer kills M1-L. |
| M2-H | [MMAE2026](#mmae2026) | Reconstruction plus latent alignment | I combine semantic NetFlow masking with causal same-event targets. | 🟡 | The hybrid is not a claim by itself. It must beat M1-R, M1-L, and MMAE-NF. |
| M2-F | [CPC](#cpc), [IJEPA](#ijepa), [VJEPA](#vjepa), [TSJEPA](#tsjepa), [LeNEPA](#lenepa), [HEPA](#hepa), [NextLat](#nextlat), [CausalJEPA](#causaljepa), [NetFlowGen](#netflowgen) | Predictive and horizon-conditioned latent SSL | I target a later completed flow touching either anchor endpoint, encode its causal prefix, then test external low-label transfer. | 🟡 | Generic future prediction is prior art. I need a matched generic future-target control. |
| M3-Ego | [CMESCrossFlow2026](#cmescrossflow2026), [MMAE2026](#mmae2026), [GraphIDS](#graphids), [VanLangendonckGraphFM](#vanlangendonckgraphfm) | Cross-flow context and topology | I use a fixed past-only endpoint history without graph message passing or identities. | 🟡 | This is a system check. It survives only if endpoint context beats matched non-endpoint context. |
| M4-Rel | [CMESCrossFlow2026](#cmescrossflow2026) | Learned cross-flow relation bias | My 16 types use directed endpoint equality under a causal mask only. | ❌ | Relation bias is prior art. It counts only inside the full system. |
| M5-Hier | [netFound](#netfound), [MM4flow](#mm4flow), [MMAE2026](#mmae2026), [FlowletFormer](#flowletformer) | Multi-level traffic representation | I summarize the same completed-flow history at several time scales. | 🟡 | Hierarchy is prior art. I run it only after measured truncation failure. |
| X1-Distill | [ConMD](#conmd), [CESNETModels](#cesnetmodels), [YaTC](#yatc) | Packet knowledge supervises traffic representations | I require exact paired packet-teacher and NetFlow-only student records. | 🟡 | Packet and flow distillation are adjacent prior art. X1 remains conditional on safe pairing. |

## Traffic and IDS papers

### FlowTransformer2024

L. D. Manocchio *et al.*, “FlowTransformer: A Transformer Framework for
Flow-based Network Intrusion Detection Systems,” *Expert Systems with
Applications*, 2024, https://doi.org/10.1016/j.eswa.2023.122564. Code:
https://github.com/liamdm/FlowTransformer. **Use:** M0 Base is my only
FlowTransformer-style baseline. **Limit:** supervised benchmark framework, not
unlabelled NetFlow SSL or deployment transfer.

### HistoricalPrototype828582d

keys-i, “added script to get all datasets,” commit `828582d0a2b6a8f8fbf090dae84328490563629f`, 21 May 2026 UTC, https://github.com/keys-i/FlowIDS/commit/828582d0a2b6a8f8fbf090dae84328490563629f. **Use:** I use it only to understand the old global eight-flow, bidirectional prototype. **Limit:** it has no fixed split, checkpoint, or result.

### EGraphSAGE

W. W. Lo *et al.*, “E-GraphSAGE,” *NOMS*, 2022, https://doi.org/10.1109/NOMS54207.2022.9789878. Code: https://github.com/waimorris/E-GraphSAGE. **Use:** supervised graph comparator. **Limit:** graph edge classification, not causal sequence SSL.

### AnomalE

E. Caville *et al.*, “Anomal-E,” *Knowledge-Based Systems*, 2022, https://doi.org/10.1016/j.knosys.2022.110030. Code: https://github.com/waimorris/Anomal-E. **Use:** graph SSL comparator. **Limit:** not a Transformer foundation model for unlabelled NetFlow.

### NFStandardFeatures

M. Sarhan *et al.*, “NetFlow Datasets for Machine Learning-Based NIDS,” 2020, https://doi.org/10.1007/978-3-030-72802-1_9; “Towards a Standard Feature Set,” 2022, https://doi.org/10.1007/s11036-021-01843-0. **Use:** feature lineage. **Limit:** no SSL or temporal-split evidence.

### NF3Temporal

M. Luay *et al.*, “Temporal Analysis of NetFlow Datasets,” arXiv:2503.04404, 2025; “Time Matters,” *IEEE Access*, 2026, https://doi.org/10.1109/ACCESS.2026.3688204. **Use:** timestamps and four v3 datasets. **Limit:** no deployment-safe split policy.

### MMAE2026

X. Liu *et al.*, “Mean Masked Autoencoder with Flow-Mixing,” arXiv:2603.29537, 2026, https://arxiv.org/abs/2603.29537. Code: https://github.com/lx6c78/MMAE. **Use:** direct M1-R, M1-L, and M2-H prior art: masked packets, FlowMix, and EMA alignment. **Limit:** five-packet byte flows, not causal endpoint-ego NetFlow.

### CMESCrossFlow2026

A. Huang *et al.*, “LLM-Driven Cross-Flow Modeling,” *CMES*, 2026, https://doi.org/10.32604/cmes.2026.083972. Data: https://github.com/liliMpro/source_dataset. **Use:** direct prior art for cross-flow organization and four-bit relation bias. **Limit:** bidirectional supervised LLM model with identities, ports, protocol, and feature similarity.

### FlowSequenceBERT

L. G. Nguyen and K. Watabe, “Flow Sequence and BERT,” *IEEE ICC*, 2023, https://doi.org/10.1109/ICC45041.2023.10279335. **Use:** flow-sequence and domain-adaptation precedent. **Limit:** no official artifact found.

### NetFlowGen

J. Zhou *et al.*, “NetFlowGen,” arXiv:2412.20635, 2024, https://arxiv.org/abs/2412.20635. **Use:** generative NetFlow pretraining comparator. **Limit:** no official code or checkpoint found.

### NEGSC

R. Xu *et al.*, “Applying SSL to NIDS for Network Flows with GNN,” *Computer Networks*, 2024, https://doi.org/10.1016/j.comnet.2024.110356. Code: https://github.com/renj-xu/NEGSC. **Use:** graph-topology SSL comparator. **Limit:** not temporal endpoint-ego sequences.

### GraphIDS

L. Guerra *et al.*, “Self-Supervised Learning of Graph Representations for NID,” *NeurIPS*, 2025, https://arxiv.org/abs/2509.16625. Code: https://github.com/lorenzo9uerra/GraphIDS. **Use:** masked traffic and graph-context comparator. **Limit:** reconstruction-error anomaly scoring does not prove causal cross-network transfer.

### VanLangendonckGraphFM

L. Van Langendonck *et al.*, “Towards a Graph-Based Foundation Model,” 2024, https://arxiv.org/abs/2409.08111. **Use:** dynamic graph SSL comparator. **Limit:** no released large checkpoint found.

### GNNet

`GNNet graph foundation model` has no unique paper, DOI, or official repository. **Limit:** I will not compare or reproduce an unidentified method.

### TimestampGraphContrastive

J. Dai *et al.*, “Timestamp-Aware Spatio-Temporal Graph Contrastive Learning,” arXiv:2606.17109, 2026, https://arxiv.org/abs/2606.17109. Code: https://github.com/Rory6235/STG-NIDS. **Use:** temporal graph comparator. **Limit:** causal service evidence is absent.

### TSDSGNN

H. Bai *et al.*, “SSL IDS Based on Dynamic Spatiotemporal Graph,” *Applied Intelligence*, 2026, https://doi.org/10.1007/s10489-026-07371-2. **Limit:** accessible primary detail is insufficient for a planned comparison.

### TrafficMAE

W. Zhu *et al.*, “TrafficMAE,” *ICT Express*, 2025, https://doi.org/10.1016/j.icte.2025.11.004. **Use:** packet/session MAE comparator. **Limit:** no official artifact found; not deployable NetFlow IDS.

### netFound

S. Beltiukov *et al.*, “netFound,” arXiv:2310.17025, revised 2026, https://arxiv.org/abs/2310.17025. Code: https://github.com/SNL-UCSB/netFound; weights: https://huggingface.co/snlucsb. **Use:** hierarchy and large-scale traffic pretraining prior art. **Limit:** different tokens and tasks.

### Lens

X. Li *et al.*, “Lens,” arXiv:2402.03646, https://arxiv.org/abs/2402.03646. **Use:** packet/flow masked-span context. **Limit:** no verified reusable artifact or independent transfer evidence.

### NetGPT

X. Meng *et al.*, “NetGPT,” arXiv:2304.09513, https://arxiv.org/abs/2304.09513. **Use:** causal packet/flow generation comparator. **Limit:** no official artifact found.

### TrafficGPT

J. Qu *et al.*, “TrafficGPT,” arXiv:2403.05822, 2024, https://arxiv.org/abs/2403.05822. Checkpoint: https://huggingface.co/LaBackDoor/trafficgpt. **Limit:** packet traffic and benchmark pretraining, not independent IDS transfer.

### ETBERT

X. Lin *et al.*, “ET-BERT,” *WWW*, 2022, https://doi.org/10.1145/3485447.3512217. Code and checkpoint: https://github.com/linwhitehat/ET-BERT. **Use:** reusable packet SSL comparator. **Limit:** encrypted packet/burst classification.

### YaTC

R. Zhao *et al.*, “Yet Another Traffic Classifier,” *AAAI*, 2023, https://doi.org/10.1609/aaai.v37i4.25674. Code and checkpoint: https://github.com/NSSL-SJTU/YaTC. **Use:** conditional packet-teacher candidate. **Limit:** paired preprocessing compatibility is required before X1.

### NetMamba

T. Wang *et al.*, “NetMamba,” *ICNP*, 2024, https://arxiv.org/abs/2405.11449. Code: https://github.com/wangtz19/NetMamba. **Use:** efficiency comparator. **Limit:** not a Transformer or open-world IDS claim.

### TrafficMoE

J. Zhou *et al.*, “Traffic-MoE,” arXiv:2601.00357, 2026, https://arxiv.org/abs/2601.00357. **Limit:** no official code found; sparse routing changes no current rung.

### FlowletFormer

L. Liu *et al.*, “FlowletFormer,” arXiv:2508.19924, 2025, https://arxiv.org/abs/2508.19924. **Use:** flowlet and field-context prior art. **Limit:** no official artifact found.

### MM4flow

L. Yang *et al.*, “MM4flow,” *ACM CCS*, 2025, https://doi.org/10.1145/3719027.3744804. **Use:** multimodal and scale prior art. **Limit:** packet/flow modalities, no released artifact found.

### NetVAD

D. Fürst *et al.*, “NetVAD,” arXiv:2606.01452, 2026, https://arxiv.org/abs/2606.01452. **Use:** identifier-free benign-only comparator. **Limit:** supervised calibration remains in evaluation.

### Rosetta

R. Xie *et al.*, “Rosetta,” *USENIX Security*, 2023, https://www.usenix.org/conference/usenixsecurity23/presentation/xie. Code: https://github.com/sunskyXX/Rosetta. **Limit:** TLS augmentation, not general NetFlow SSL.

### NetSSM

A. Chu *et al.*, “NetSSM,” *ACM on Networking*, 2026, https://doi.org/10.1145/3786289. **Limit:** trace generation, not IDS.

## Foundations for SSL and model design

### DenoisingAutoencoder

P. Vincent *et al.*, “Denoising Autoencoders,” *ICML*, 2008, https://doi.org/10.1145/1390156.1390294. **Use:** corrupt-and-reconstruct precedent.

### BERT

J. Devlin *et al.*, “BERT,” *NAACL-HLT*, 2019, https://doi.org/10.18653/v1/N19-1423. **Limit:** bidirectional language masking, not traffic.

### GPT

A. Radford *et al.*, “Improving Language Understanding by Generative Pre-Training,” 2018, https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf. **Limit:** causal language model, not traffic.

### CPC

A. van den Oord *et al.*, “Contrastive Predictive Coding,” arXiv:1807.03748, 2018, https://arxiv.org/abs/1807.03748. **Use:** future-prediction precedent.

### SimCLR

T. Chen *et al.*, “SimCLR,” *ICML*, 2020, https://proceedings.mlr.press/v119/chen20j.html. **Limit:** visual contrastive SSL.

### BYOL

J.-B. Grill *et al.*, “BYOL,” *NeurIPS*, 2020, https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html. **Use:** EMA teacher/student precedent.

### DINO

M. Caron *et al.*, “DINO,” *ICCV*, 2021, https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html. **Limit:** vision SSL.

### Barlow

J. Zbontar *et al.*, “Barlow Twins,” *ICML*, 2021, https://proceedings.mlr.press/v139/zbontar21a.html. **Use:** redundancy-reduction control.

### VICReg

A. Bardes *et al.*, “VICReg,” *ICLR*, 2022, https://openreview.net/forum?id=xm6YD62D1Ub. **Use:** variance and covariance anti-collapse reference.

### MAE

K. He *et al.*, “Masked Autoencoders,” *CVPR*, 2022, https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html. **Use:** masked reconstruction precedent.

### data2vec

A. Baevski *et al.*, “data2vec,” *ICML*, 2022, https://proceedings.mlr.press/v162/baevski22a.html. **Use:** M1-L uses its masked-student, full-input EMA-teacher pattern. **Limit:** same-event target, not future prediction or NetFlow evidence.

### IJEPA

A. Assran *et al.*, “I-JEPA,” *CVPR*, 2023, https://arxiv.org/abs/2301.08243. Code: https://github.com/facebookresearch/ijepa. **Use:** masked joint-embedding precedent. **Limit:** same-image target, not causal future events.

### VJEPA

A. Bardes *et al.*, “V-JEPA,” arXiv:2404.08471, 2024, https://arxiv.org/abs/2404.08471. Code: https://github.com/facebookresearch/jepa. **Use:** video feature-prediction precedent. **Limit:** not NetFlow or causal serving.

### TSJEPA

S. Ennadir *et al.*, “Joint Embeddings Go Temporal,” arXiv:2509.25449, 2025, https://arxiv.org/abs/2509.25449. Code: https://github.com/Sennadir/TS_JEPA. **Limit:** masked univariate time series, not causal NetFlow.

### LeNEPA

A. Chemeris *et al.*, “LeNEPA,” arXiv:2607.00958, 2026, https://arxiv.org/abs/2607.00958. Code: https://github.com/langotime/lenepa-milets-2026. **Use:** direct causal next-latent prior art. **Limit:** no endpoint-incident NetFlow target.

### HEPA

J. Petersen *et al.*, “HEPA: A Self-Supervised Horizon-Conditioned Event
Predictive Architecture for Time Series,” arXiv:2605.11130, 2026,
https://arxiv.org/abs/2605.11130. Code: https://github.com/Forgis-Labs/hepa.
**Use:** generic horizon-conditioned future-latent SSL is prior art. **Limit:**
HEPA changes M2-F into an endpoint-incident versus generic-future control.

### NextLat

J. Teoh *et al.*, “Next-Latent Prediction Transformers Learn Compact World
Models,” arXiv:2511.05963, 2025, https://arxiv.org/abs/2511.05963. Code:
https://github.com/JaydenTeoh/NextLat. **Use:** generic next-latent prediction is
prior art. **Limit:** it does not establish endpoint-incident NetFlow transfer.

### CausalJEPA

H. Nam *et al.*, “Causal-JEPA: Learning World Models through Object-Level
Latent Masking,” arXiv:2602.11389, revised 28 May 2026,
https://arxiv.org/abs/2602.11389. Code:
https://github.com/galilai-group/cjepa. **Use:** causal masked latent prediction
precedent. **Limit:** object-centric world modelling, not the NetFlow target.

### TalkLikeAPacket

S. Mayhoub *et al.*, “Talk Like a Packet: Rethinking Network Traffic Analysis
with Transformer Foundation Models,” arXiv:2602.06636, 2026,
https://arxiv.org/abs/2602.06636. **Use:** packet-language pretraining overview.
**Limit:** the arXiv record links no model artifact and changes no model rung.

## Related tabular, temporal, graph, and state-space work

### VIME

J. Yoon *et al.*, “VIME,” *NeurIPS*, 2020, https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html. **Use:** tabular masking control.

### SCARF

P. Bahri *et al.*, “SCARF,” *ICLR*, 2022, https://openreview.net/forum?id=CuVqYB0kZci. **Use:** feature-corruption contrastive control.

### FTTransformer

Y. Gorishniy *et al.*, “Revisiting Deep Learning Models for Tabular Data,” *NeurIPS*, 2021, https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html. **Use:** supervised tabular comparator.

### STRATS

S. Tipirneni and C. Reddy, “STraTS,” arXiv:2107.14293, 2021, https://arxiv.org/abs/2107.14293. **Use:** missingness and time-gap encoding precedent.

### TS2Vec

Z. Yue *et al.*, “TS2Vec,” *AAAI*, 2022, https://ojs.aaai.org/index.php/AAAI/article/view/20881. **Use:** hierarchical time-series comparator.

### TimeMAE

M. Cheng *et al.*, “TimeMAE,” *WSDM*, 2026, https://doi.org/10.1145/3773966.3778007. **Use:** decoupled masking and momentum regression precedent.

### TiMAE

Z. Li *et al.*, “Ti-MAE,” arXiv:2301.08871, 2023, https://arxiv.org/abs/2301.08871. **Use:** masked temporal comparator.

### SetTransformer

J. Lee *et al.*, “Set Transformer,” *ICML*, 2019, https://proceedings.mlr.press/v97/lee19d.html. **Use:** set-attention reference.

### Graphormer

C. Ying *et al.*, “Graphormer,” *NeurIPS*, 2021, https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html. **Use:** structural-attention-bias reference.

### DGI

P. Veličković *et al.*, “Deep Graph Infomax,” *ICLR*, 2019, https://openreview.net/forum?id=rklz9iAcKQ. **Use:** graph SSL comparator.

### GraphMAE2

W. Hou *et al.*, “GraphMAE2,” *WWW*, 2024, https://dl.acm.org/doi/10.1145/3589334.3645451. **Use:** graph masked-autoencoding reference.

### TGN

E. Rossi *et al.*, “Temporal Graph Networks,” arXiv:2006.10637, 2020, https://arxiv.org/abs/2006.10637. **Use:** causal event-memory comparator.

### Mamba

A. Gu and T. Dao, “Mamba,” arXiv:2312.00752, 2023, https://arxiv.org/abs/2312.00752. **Use:** state-space efficiency comparator.

### Chronos

A. F. Ansari *et al.*, “Chronos,” arXiv:2403.07815, 2024, https://arxiv.org/abs/2403.07815. Code: https://github.com/amazon-science/chronos-forecasting. **Limit:** forecasting, not IDS.

## Adaptation and anomaly detection

### DAPT

S. Gururangan *et al.*, “Don’t Stop Pretraining,” *ACL*, 2020, https://aclanthology.org/2020.acl-main.740/. **Use:** adaptation precedent.

### LoRA

E. Hu *et al.*, “LoRA,” *ICLR*, 2022, https://openreview.net/forum?id=nZeVKeeFYf9. **Use:** efficient adaptation comparator.

### DeepSVDD

L. Ruff *et al.*, “Deep One-Class Classification,” *ICML*, 2018, https://proceedings.mlr.press/v80/ruff18a.html. **Use:** one-class control.

### EnergyOOD

W. Liu *et al.*, “Energy-based OOD Detection,” *NeurIPS*, 2020, https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html. **Use:** OOD control.

### ConMD

X. Lian *et al.*, “Contextual Masking Distillation for Network Traffic Anomaly
Detection,” *IEEE Transactions on Information Forensics and Security*, 2026,
https://doi.org/10.1109/TIFS.2026.3655514. Code:
https://github.com/ikun0124/ConMD. **Use:** packet and flow distillation makes
X1 adjacent work. **Limit:** I still need exact paired data and compatible
preprocessing.

### CESNETModels

CESNET, “cesnet-models,” official repository, https://github.com/CESNET/cesnet-models. **Use:** maintained traffic-model artifacts strengthen the X1 comparison set. **Limit:** an artifact is not automatic teacher compatibility.

## Evaluation methods

### NTFMReview2026

R. Pérez-Jove *et al.*, “Network Traffic Foundation Models: A Systematic Review,” *Computer Networks*, 2026, https://doi.org/10.1016/j.comnet.2026.111998. **Limit:** maps the field but cannot establish a novelty claim alone.

### Dedup

K. Lee *et al.*, “Deduplicating Training Data,” *ACL*, 2022, https://aclanthology.org/2022.acl-long.577/. **Use:** duplicate-control rationale.

### ComputeScaling

J. Kaplan *et al.*, arXiv:2001.08361, 2020, https://arxiv.org/abs/2001.08361; J. Hoffmann *et al.*, arXiv:2203.15556, 2022, https://arxiv.org/abs/2203.15556. **Use:** measure data, model, FLOPs, and transfer, not pretraining loss alone.

### NetBench

C. Qian *et al.*, “NetBench,” arXiv:2403.10319, 2024, https://arxiv.org/abs/2403.10319. **Limit:** packet benchmark, not an independent NetFlow deployment benchmark.

### ShortcutLearning

R. Geirhos *et al.*, “Shortcut Learning,” *Nature Machine Intelligence*, 2020, https://doi.org/10.1038/s42256-020-00257-z. **Use:** shortcut-risk rationale.

### PRvsROC

J. Davis and M. Goadrich, *ICML*, 2006, https://dl.acm.org/doi/10.1145/1143844.1143874; T. Saito and M. Rehmsmeier, *PLOS ONE*, 2015, https://doi.org/10.1371/journal.pone.0118432. **Use:** report PR, ROC, prevalence, and alert volume.

### Calibration

C. Guo *et al.*, “On Calibration,” *ICML*, 2017, https://proceedings.mlr.press/v70/guo17a.html. **Use:** calibrate on held-out relevant data.

### StatisticalTests

J. Demšar, “Statistical Comparisons of Classifiers,” *JMLR*, 2006, https://jmlr.org/papers/v7/demsar06a.html. **Use:** paired repeated comparisons.

## Standards and datasets

### IPFIXRFC7011

B. Claise *et al.*, RFC 7011, 2013, https://doi.org/10.17487/RFC7011. **Use:** IPFIX protocol standard.

### IPFIXRFC7012

B. Claise and B. Trammell, RFC 7012, 2013, https://doi.org/10.17487/RFC7012. **Use:** IPFIX information model.

### NetFlowV9RFC3954

Cisco, RFC 3954, 2004, https://doi.org/10.17487/RFC3954. **Use:** NetFlow v9 format.

### NF3UQData

University of Queensland, “NetFlow Datasets,” https://staff.itee.uq.edu.au/marius/NIDS_datasets/. **Use:** source of the four NF3 benchmark inputs. **Limit:** converted lineage supports benchmark comparison, not independent operational claims.

### CTU13

S. García *et al.*, “An Empirical Comparison of Botnet Detection Methods,” *Computers & Security*, 2014, https://doi.org/10.1016/j.cose.2014.05.011. Data: https://stratosphere-ips.squarespace.com/datasets-ctu13. **Use:** scenario-held-out flow benchmark. **Limit:** schema and release details differ across scenarios.

### CESNETTLSYear22

K. Hynek *et al.*, “CESNET-TLS-Year22,” *Scientific Data*, 2024, https://doi.org/10.1038/s41597-024-03927-4. Data: https://github.com/CESNET/cesnet-datazoo. **Use:** long-span TLS flow probe. **Limit:** label fields cannot be inputs; missing probe identity blocks endpoint histories.

### CICIot2022

S. Dadkhah *et al.*, “CIC IoT Dataset 2022,” *PST*, 2022, https://doi.org/10.1109/PST55820.2022.9851966. Data: https://www.unb.ca/cic/datasets/iotdataset-2022.html. **Limit:** device-disjoint claims require verified capture groups.

### UGR16

G. Maciá-Fernández *et al.*, “UGR’16,” *Computers & Security*, 2018, https://doi.org/10.1016/j.cose.2018.03.004. Data: https://nesg.ugr.es/nesg-ugr16/. **Limit:** injected attacks and metadata gaps constrain chronology claims.

### LITNET2020

R. Damasevicius *et al.*, “LITNET-2020,” *Electronics*, 2020, https://doi.org/10.3390/electronics9050800. Data: https://dataset.litnet.lt/data.php. **Limit:** flow/exporter chronology must be checked before use.

### CICDocs

Canadian Institute for Cybersecurity, “Datasets,” https://www.unb.ca/cic/datasets/. **Use:** dataset-specific collection and label documentation.

### MAWI

WIDE Project, “MAWI Working Group Traffic Archive,” https://mawi.wide.ad.jp/mawi/. **Use:** unlabelled SSL, drift, and alert-volume exploration. **Limit:** no stable cross-trace identity or IDS ground truth; respect privacy and reuse rules. MAWILab stopped updates in December 2024.

## Work left out

[XION](https://arxiv.org/abs/2608.26831) is not in the core table: its dependency-based anomaly score changes no SSL rung. I also leave out the unidentified contextualised-NetFlow review. Unsourced summaries and marketing pages are not novelty evidence.
