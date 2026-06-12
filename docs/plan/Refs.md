# Research evidence and novelty ledger

**Cutoff:** 2026-08-19. This is a finite audit of original papers, official
artifacts, standards, datasets, and one explicitly labeled systematic
review—not an exhaustive review. Search coverage includes IEEE Xplore, ACM DL, SpringerLink,
ScienceDirect, USENIX, arXiv, official standards/data/repositories, and
backward/forward citation trails. Query families: `NetFlow AND (SSL OR foundation OR transformer
OR masked)`; `network traffic AND (MAE OR teacher-student OR EMA OR cross-flow
OR relation)`; `IDS AND (temporal graph OR domain transfer OR few-shot OR
cross-dataset)`; and `time series AND (JEPA OR joint embedding OR
next-latent)`. Venue indexing and repository links were checked where stated;
an `UNVERIFIED` field means no usable primary-method record was verified here.

**Ledger convention:** the citation states publication status; each following
sentence records verified modality, objective, context, data/split, scale,
result, artifact, imported evidence, or limitation as applicable. Any split,
code, data, parameter, compute, or result field not stated is `UNVERIFIED` and
cannot support a project claim.

## Prior-art and reproduction boundaries

### Closest-work matrix

| Claim | Closest record(s) | Exact overlap | Material difference | Reproducibility | Verdict |
|---|---|---|---|---|---|
| M1-R | [MMAE2026](#mmae2026) | Generic masked traffic representation learning | MMAE reconstructs five-packet byte flows with FlowMix; M1-R masks NetFlow feature groups | Code released; reproduction UNVERIFIED | Generic masked-flow novelty rejected |
| M1-L | [MMAE2026](#mmae2026), [data2vec](#data2vec) | Corrupted student aligned to an unmasked EMA teacher | MMAE aligns reconstructed byte-flow tokens; M1-L targets a causal NetFlow event state | MMAE code released | EMA/latent-target novelty rejected |
| M2-H | [MMAE2026](#mmae2026) | Reconstruction plus EMA latent alignment | Different modality, corruption, context, and target | MMAE code released | Generic hybrid-objective novelty rejected |
| M2-F | [CPC](#cpc), [IJEPA](#ijepa), [VJEPA](#vjepa), [TSJEPA](#tsjepa), [LeNEPA](#lenepa), [NetFlowGen](#netflowgen) | Predictive latent learning, masked joint embedding, temporal feature prediction, or pretrained NetFlow dynamics; LeNEPA directly covers causal next-latent prediction | The proposed target is a later completed flow incident to either anchor endpoint, encoded with its own causal prefix and tested under external low-label transfer | I-JEPA, V-JEPA, TS-JEPA, and LeNEPA code released but reproduction UNVERIFIED; NetFlowGen code/checkpoint UNVERIFIED | Generic JEPA-family and next-latent novelty rejected; the endpoint-incident causal NetFlow formulation is provisional, not established novelty |
| M3-Ego | [CMESCrossFlow2026](#cmescrossflow2026), [MMAE2026](#mmae2026), [VanLangendonckGraphFM](#vanlangendonckgraphfm) | Cross-flow organization, corruption, or endpoint topology | CMES is rule-grouped/bidirectional; MMAE uses a support flow as a patch repository; the graph FM is a different dynamic graph unit | CMES model code UNVERIFIED; graph checkpoint UNVERIFIED | System component only, defensible after endpoint-defined past-only causal ablations pass |
| M4-Rel | [CMESCrossFlow2026](#cmescrossflow2026) | Four-bit/16-type learned cross-flow attention bias | Proposed bits encode directed endpoint equality under a causal mask | Model code UNVERIFIED | No standalone relation-bias novelty; system component only |
| M5-Hier | [netFound](#netfound), [MM4flow](#mm4flow), [MMAE2026](#mmae2026) | Multi-granularity or multimodal traffic representation | Different units and hierarchy | netFound/MMAE artifacts available; MM4flow reproduction UNVERIFIED | Hierarchy novelty rejected |
| X1-Distill | Generic distillation plus [YaTC](#yatc), [netFound](#netfound), and [MM4flow](#mm4flow) | Packet knowledge transferred into traffic representations | Exact packet-teacher to NetFlow-only student claim not verified in this search; YaTC is a packet-view comparator and conditional teacher candidate, not an interchangeable teacher | Requires renewed search, paired data, and verified preprocessing/checkpoint compatibility | Provisional only |

## Requested traffic/IDS records

### FlowTransformer2024

L. D. Manocchio, S. Layeghy, W. W. Lo, G. K. Kulatilleke, M. Sarhan, and
M. Portmann, “FlowTransformer: A Transformer Framework for Flow-based Network
Intrusion Detection Systems,” *Expert Systems with Applications*, vol. 241,
122564, 2024, https://doi.org/10.1016/j.eswa.2023.122564. Preprint:
https://arxiv.org/abs/2304.14746. Official code:
https://github.com/liamdm/FlowTransformer. It is a supervised configurable
flow-sequence classifier evaluated on NSL-KDD, UNSW-NB15, and
CSE-CIC-IDS2018. It is a required historical baseline, not SSL, a reusable
checkpoint, or a foundation model; its published benchmark protocol does not
establish deployment transfer.

### HistoricalPrototype828582d

keys-i, “added script to get all datasets,” repository commit `828582d0a2b6a8f8fbf090dae84328490563629f`, 22 May 2026,
https://github.com/keys-i/FlowIDS/commit/828582d0a2b6a8f8fbf090dae84328490563629f.
Repository artifact. The snapshot configures fixed eight-flow windows over each
source globally sorted by `FLOW_START_MILLISECONDS`, no positional signal, and
a bidirectional PyTorch `TransformerEncoder`; its heads make next-flow
multitask categorical and numeric predictions. It is forensic historical
evidence only: no eligible checkpoint, immutable data/split artifact, or
reproducible result accompanies that commit, so it is not a baseline result or
a precursor foundation model.

### EGraphSAGE

W. W. Lo, S. Layeghy, M. Sarhan, M. Gallagher, and M. Portmann,
“E-GraphSAGE: A Graph Neural Network Based Intrusion Detection System for
IoT,” *NOMS 2022*, pp. 1–9, 2022,
https://doi.org/10.1109/NOMS54207.2022.9789878. Preprint:
https://arxiv.org/abs/2103.16329. Official code:
https://github.com/waimorris/E-GraphSAGE. Endpoint IPs form nodes and
feature-bearing flows form edges for supervised edge classification. It is a
graph challenger, not self-supervised, temporal, causal, or a foundation model.

### AnomalE

E. Caville, W. W. Lo, S. Layeghy, and M. Portmann, “Anomal-E: A
Self-Supervised Network Intrusion Detection System Based on Graph Neural
Networks,” *Knowledge-Based Systems*, vol. 258, 110030, 2022,
https://doi.org/10.1016/j.knosys.2022.110030. Preprint:
https://arxiv.org/abs/2207.06819. Official code:
https://github.com/waimorris/Anomal-E. Modified Deep Graph Infomax trains an
E-GraphSAGE edge encoder before a separate unsupervised outlier detector. It is
graph SSL, not a causal Transformer or a pretrained NetFlow foundation model;
its graph and threshold assumptions remain benchmark-specific. See
[GraphIDS](#graphids) Appendix A for its public-implementation caveat.

### NFStandardFeatures

M. Sarhan, S. Layeghy, N. Moustafa, and M. Portmann, “NetFlow Datasets for
Machine Learning-Based Network Intrusion Detection Systems,” *BDTA 2020*,
https://doi.org/10.1007/978-3-030-72802-1_9; preprint:
https://arxiv.org/abs/2011.09144. M. Sarhan, S. Layeghy, and M. Portmann,
“Towards a Standard Feature Set for Network Intrusion Detection System
Datasets,” *Mobile Networks and Applications*, 2022,
https://doi.org/10.1007/s11036-021-01843-0. These papers establish the UQ
NetFlow conversion and standard-feature lineage. They do not establish
temporal splits, SSL, or foundation-model evidence; conversions of one capture
remain one lineage.

### NF3Temporal

M. Luay, S. Layeghy, S. Hosseininoorbin, M. Sarhan, N. Moustafa, and
M. Portmann, “Temporal Analysis of NetFlow Datasets for Network Intrusion
Detection Systems,” arXiv:2503.04404v2, 2025,
https://arxiv.org/abs/2503.04404. M. Luay, S. Layeghy, N. Noorbin,
M. Sarhan, G. Kulatilleke, N. Moustafa, and M. Portmann, “Time Matters:
Temporal NetFlow Features for ML-Based Network Intrusion Detection,” *IEEE
Access*, vol. 14, pp. 66899–66913, 2026,
https://doi.org/10.1109/ACCESS.2026.3688204. The official UQ collection has
four v3 datasets—NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3, and
NF-CSE-CIC-IDS2018-v3—with 43 v2 and ten temporal features:
https://staff.itee.uq.edu.au/marius/NIDS_datasets/. It supplies timestamps and
directional IAT features, not a deployment-safe split policy. There is no
official `NF-UQ-NIDS-v3` merged dataset.

### MMAE2026

X. Liu, X. Fu, F. Huang, and L. Zhang, “Mean Masked Autoencoder with
Flow-Mixing for Encrypted Traffic Classification,” arXiv:2603.29537v1,
31 Mar. 2026, https://arxiv.org/abs/2603.29537. Author code:
https://github.com/lx6c78/MMAE; reproduction **UNVERIFIED**. The unit is a
SplitCap five-tuple flow represented by the first five packets × 320 bytes
(80 header + 240 payload) = 1,600 bytes; IPs and ports are anonymized.
Twenty-seven side statistics guide matching and masking. A batch-paired support
flow supplies replacement patches, while the EMA teacher receives the unmasked
main-flow sequence. FlowMix uses 70% random plus up to 20% hard patches; losses
combine reconstruction, mask prediction, and cosine EMA alignment
(0.96→0.99). The seven-block encoder/two-block decoder is width 256 and 8.65M
pretraining parameters. The paper states six pretraining and seven evaluation
datasets, an 8:1:1 split without temporal/entity detail, and 150k iterations at
batch 128 on one RTX 4080 (24.66 h/18 GB); weighted F1 0.9434–1.0000 is
author-reported. It anticipates generic masked teacher–student traffic learning,
not causal endpoint-ego NetFlow modeling.

### CMESCrossFlow2026

A. Huang, S. Zhang, H. Zhu, X. Fan, and H. Zhou, “LLM-Driven Cross-Flow
Modeling for Network Attack Traffic Detection,” *Computer Modeling in
Engineering & Sciences*, vol. 148, no. 1, p. 50, 2026,
https://doi.org/10.32604/cmes.2026.083972. Received 14 Apr., accepted 5 Jun.,
and issue-published 27 Jul. 2026. Author data:
https://github.com/liliMpro/source_dataset; model code **UNVERIFIED**. Flow
statistics undergo RF selection/correlation, feature sorting/grouping, and
cross-group sampling. Four bits—endpoint, destination port, protocol, and
feature similarity—form 16 relation types. Dual MLP/Transformer embeddings
feed a bidirectional frozen 8B LLM with learned relation bias; the causal mask
is removed and only additions/head train under supervision. Tests cover lab
DDoS, CICDDoS2019, and encrypted traffic, including 8:2 stratified in-domain,
two cross-dataset directions, and held-out-type “zero-shot-like” settings.
Cross-domain accuracy 0.9418/0.9496 is author-reported; AUPRC, alert load, and
causal streaming are not evaluated. It directly blocks standalone cross-flow
relation-bias novelty and partly overlaps ego-context construction.

### FlowSequenceBERT

L. G. Nguyen and K. Watabe, “A Method for Network Intrusion Detection Using
Flow Sequence and BERT Framework,” *IEEE ICC 2023*, pp. 3006–3011,
https://doi.org/10.1109/ICC45041.2023.10279335; preprint:
https://arxiv.org/abs/2310.17127. It establishes BERT flow-sequence NIDS/domain
adaptation as prior art. No official code was verified, so reproduction and
split discipline remain **UNVERIFIED**.

### NetFlowGen

J. Zhou, W. Kim, Z. Xu, A. M. Rush, and M. Yu, “NetFlowGen: Leveraging
Generative Pre-training for Network Traffic Dynamics,” arXiv:2412.20635,
2024, https://arxiv.org/abs/2412.20635. It pretrains generatively on unlabeled
NetFlow for traffic dynamics and few-label downstream adaptation. No official
code/checkpoint was verified; the preprint and fixed schema do not establish a
portable operational foundation model.

### NEGSC

R. Xu, G. Wu, W. Wang, X. Gao, A. He, and Z. Zhang, “Applying
Self-Supervised Learning to Network Intrusion Detection for Network Flows with
Graph Neural Network,” *Computer Networks*, vol. 244, 110356, 2024,
https://doi.org/10.1016/j.comnet.2024.110356. Preprint:
https://arxiv.org/abs/2403.01501. Official code:
https://github.com/renj-xu/NEGSC. NEGAT plus structured generated-subgraph
contrast learns local NetFlow topology for binary/multiclass IDS. It is not
temporal endpoint-ego sequence modeling, and its notebook artifact and
benchmark protocol require reproduction.

### GraphIDS

L. Guerra, T. Chapuis, G. Duc, P. Mozharovskyi, and V.-T. Nguyen,
“Self-Supervised Learning of Graph Representations for Network Intrusion
Detection,” *NeurIPS 2025*, https://arxiv.org/abs/2509.16625. Official paper:
https://papers.nips.cc/paper_files/paper/2025/hash/9ddb13ae9150f99298065d889f951014-Abstract-Conference.html.
Official code: https://github.com/lorenzo9uerra/GraphIDS. E-GraphSAGE local
flow embeddings feed a Transformer masked autoencoder whose reconstruction
error is an anomaly score. Its reported PR-AUC up to 99.98% is author evidence
under benchmark protocols, not proof of causal cross-network transfer. Appendix
A narrowly reports that the inspected public Anomal-E code used attack labels
in target encoding; it is an implementation-specific leakage finding, not a
claim that every Anomal-E experiment is invalid.

### VanLangendonckGraphFM

L. Van Langendonck, I. Castell-Uroz, and P. Barlet-Ros, “Towards a
Graph-Based Foundation Model for Network Traffic Analysis,” 2024,
https://arxiv.org/abs/2409.08111; workshop record:
https://doi.org/10.1145/3694811.3697817. A dynamic spatio-temporal line graph
uses SSL link prediction before few-shot intrusion, traffic, and botnet tasks;
the reported average gain over scratch is 6.87%. The authors use two small
unlabeled sources and frame this as groundwork; no large released checkpoint
or operational foundation-model evidence was verified.

### GNNet

`GNNet graph foundation model` is **UNRESOLVED**: the supplied name does not
identify a unique traffic model, paper, DOI, or official repository. It is not
a synonym assigned here to [VanLangendonckGraphFM](#vanlangendonckgraphfm).
No reproduction or comparison is permitted until the exact primary artifact
is identified.

### TimestampGraphContrastive

J. Dai, G. Wu, J. Li, W. Wang, A. He, and X. Xiao, “Timestamp-Aware
Spatio-Temporal Graph Contrastive Learning for Network Intrusion Detection,”
arXiv:2606.17109, 2026, https://arxiv.org/abs/2606.17109. Official code:
https://github.com/Rory6235/STG-NIDS. Timestamped temporal graphs feed an
E-GraphSAGE/LSTM encoder with temporal, spatial, and feature contrastive
losses. This is a June 2026 preprint; causal serving and reported gains remain
unreproduced here.

### TSDSGNN

H. Bai, L. Chen, and Q. Dai, “Self-Supervised Intrusion Detection Algorithm
Based on Dynamic Spatiotemporal Graph,” *Applied Intelligence*, vol. 56,
no. 10, art. 344, 2026, https://doi.org/10.1007/s10489-026-07371-2.
Publisher-method access was unavailable during this audit; detailed method,
data, and result claims remain **UNVERIFIED**.

### TrafficMAE

W. Zhu, H. Zhang, C. Pei, and J. Li, “TrafficMAE: A Network Traffic
Classification Model Based on Masked Autoencoder,” *ICT Express*, online
13 Nov. 2025, https://doi.org/10.1016/j.icte.2025.11.004. Eight-packet
Session Videos feed a ViT MAE with reconstruction and direction prediction;
fine-tuning fuses statistical tokens. The author reports 5.36M parameters and
F1 0.9633–0.9976 across five tasks. No official code/checkpoint was verified;
it is packet/session classification rather than deployable NetFlow IDS.

### netFound

S. Beltiukov *et al.*, “netFound: Principled Design for Network Foundation
Models,” arXiv:2310.17025v5, revised 12 May 2026,
https://arxiv.org/abs/2310.17025. Official code:
https://github.com/SNL-UCSB/netFound. Weights:
https://huggingface.co/snlucsb. Protocol-aware tokens, operational context,
and burst/flow hierarchical attention support masked-token SSL without payload
or IP input. The release reports 53M–663M models, a 4.2B-flow corpus, and
5,000 GPU-hours. Results are author-reported and do not establish open-world
IDS alert performance, but hierarchy and broad traffic-pretraining novelty are
clearly anticipated.

### Lens

X. Li *et al.*, “Lens: A Knowledge-Guided Foundation Model for Network
Traffic,” arXiv:2402.03646, current 2026 revision,
https://arxiv.org/abs/2402.03646. A roughly 250M T5 model uses parsed packet/
flow BBPE plus source context, knowledge-guided masked spans, and constrained
generative fine-tuning across NetBench classification/generation tasks. Code
and reusable weights were not verified; same-dataset pretraining and source
context complicate external-transfer claims.

### NetGPT

X. Meng, C. Lin, Y. Wang, and Y. Zhang, “NetGPT: Generative Pretrained
Transformer for Network Traffic,” arXiv:2304.09513,
https://arxiv.org/abs/2304.09513. GPT-2-style causal hexadecimal packet/flow
tokens, header shuffling, separators, and prompts support classification,
attack detection, and header generation. No official code/checkpoint was
verified; data scale and transfer evidence are insufficient for a current
reproducible foundation baseline.

### TrafficGPT

J. Qu, X. Ma, and J. Li, “TrafficGPT: Breaking the Token Barrier for
Efficient Long Traffic Analysis and Generation,” arXiv:2403.05822, 2024,
https://arxiv.org/abs/2403.05822. Checkpoint:
https://huggingface.co/LaBackDoor/trafficgpt. A reversible byte representation
and linear-attention causal model handle up to 12,032 tokens across 189 GB of
public traffic. It is a preprint without verified source code or independent
cross-network IDS evaluation; almost all downstream benchmark traffic is used
for pretraining.

### ETBERT

X. Lin, G. Xiong, G. Gou, Z. Li, J. Shi, and J. Yu, “ET-BERT: A
Contextualized Datagram Representation with Pre-Training Transformers for
Encrypted Traffic Classification,” *WWW 2022*, pp. 633–642,
https://doi.org/10.1145/3485447.3512217. Official code/checkpoint:
https://github.com/linwhitehat/ET-BERT. Masked-burst and same-origin burst
pretraining is legitimate reusable SSL, but the scope is encrypted
packet/burst classification rather than general NetFlow transfer.

### YaTC

R. Zhao *et al.*, “Yet Another Traffic Classifier: A Masked Autoencoder Based
Traffic Transformer with Multi-Level Flow Representation,” *AAAI 2023*,
vol. 37, no. 4, pp. 5420–5427,
https://doi.org/10.1609/aaai.v37i4.25674. Official code/checkpoint:
https://github.com/NSSL-SJTU/YaTC. MAE pretraining operates on formatted
multi-level flow matrices with packet/flow attention. Its four encrypted
classification benchmarks do not establish general IDS transfer. It is a
packet-view comparator and only a conditional X1-Distill teacher candidate: its packet
formatting, feature preprocessing, and released-checkpoint compatibility with
paired PCAP/NetFlow data must be verified before it can supervise a student.

### NetMamba

T. Wang *et al.*, “NetMamba: Efficient Network Traffic Classification via
Pre-Training Unidirectional Mamba,” *ICNP 2024*,
https://arxiv.org/abs/2405.11449. Official code:
https://github.com/wangtz19/NetMamba. Checkpoint:
https://huggingface.co/wangtz19/NetMamba. This MAE-like Mamba/SSM traffic
classifier is an efficiency comparator, not a Transformer or evidence of
open-world IDS. NetMamba+ (arXiv:2601.21792) is a separate 2026 preprint.

### TrafficMoE

J. Zhou, C. Sun, W. Fu, M. Shen, S. Yu, and Q. Xuan, “Traffic-MoE: A
Sparse Foundation Model for Network Traffic Security Analysis,”
arXiv:2601.00357v2, revised 8 Jul. 2026,
https://arxiv.org/abs/2601.00357. Sparse expert routing is reported across four
security tasks with 70.42% higher throughput and 41.39% lower latency. No
official code was verified; this is a preprint and must not be conflated with
the separate supervised `TrafficMoE` encrypted classifier (arXiv:2603.29520).

### FlowletFormer

L. Liu, R. Li, Q. Li, M. Hou, Y. Jiang, and M. Xu, “FlowletFormer:
Network Behavioral Semantic Aware Pre-Training Model for Traffic
Classification,” arXiv:2508.19924, 2025,
https://arxiv.org/abs/2508.19924. It uses IAT-derived flowlets,
protocol-aligned embeddings, and field/context pretraining tasks. No verified
official code, checkpoint, or archival venue exists; reproduction is
**UNVERIFIED**.

### MM4flow

L. Yang, L. Liu, J. Huang, Z. Liu, S. Liang, S. Fu, and Y. Wang, “MM4flow:
A Pre-trained Multi-modal Model for Versatile Network Traffic Analysis,”
*ACM CCS 2025*, pp. 1664–1678,
https://doi.org/10.1145/3719027.3744804. It separately pretrains byte-stream
and transmission-pattern modalities on author-reported traffic exceeding
70 TB, then fuses them with cross-attention for six downstream task types.
This is strong multimodal and scale prior art, but it is packet/flow traffic
rather than a NetFlow-only causal ego model; code, checkpoints, corpus access,
split discipline, and numerical reproduction remain **UNVERIFIED**.

### NetVAD

D. Fürst, P. Levi, and S. Steindl, “NetVAD: Foundation-Model Representation
Learning for Identifier-Free Unsupervised Intrusion Detection,” accepted at
IEEE ISI 2026, arXiv:2606.01452v2, revised 29 Jun. 2026,
https://arxiv.org/abs/2606.01452. A benign-only VAE models representations
from a frozen network foundation model without identifiers; ToN-IoT and
IoT-23 results include author-reported 98% micro F1 and 96% macro F1 at an
operational false-positive rate. It is a benign-only-training comparator, but
not an end-to-end zero-label comparator because reported evaluation includes
supervised calibration. It is not evidence for causal endpoint-ego
pretraining; single-packet reconnaissance failures, code availability, and
reproduction remain limitations.

### Rosetta

R. Xie *et al.*, “Rosetta: Enabling Robust TLS Encrypted Traffic
Classification in Diverse Network Environments with TCP-Aware Traffic
Augmentation,” *USENIX Security 2023*, pp. 625–642,
https://www.usenix.org/conference/usenixsecurity23/presentation/xie. Official
code: https://github.com/sunskyXX/Rosetta. TCP-aware augmentations train a
contrastive packet-length representation for TLS robustness. It is a narrow
augmentation/classification method, not a general traffic foundation model.

### NetSSM

A. Chu, X. Jiang, S. Liu, A. Bhagoji, F. Bronzino, P. Schmitt, and
N. Feamster, “NetSSM: Multi-Flow and State-Aware Network Trace Generation
Using State-Space Models,” *Proceedings of the ACM on Networking*, vol. 4,
CoNEXT 2026, https://doi.org/10.1145/3786289. Preprint:
https://arxiv.org/abs/2503.22663. It models interleaved flow/session state for
trace generation, not IDS or classification; source code/checkpoint was not
verified.

## SSL and model roots

### DenoisingAutoencoder

P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and
Composing Robust Features with Denoising Autoencoders,” *ICML 2008*,
https://doi.org/10.1145/1390156.1390294. Corrupt-and-reconstruct pretraining is
therefore established prior art; it contributes no traffic-specific novelty.

### BERT

J. Devlin *et al.*, “BERT: Pre-Training of Deep Bidirectional Transformers
for Language Understanding,” *NAACL-HLT*, 2019, doi:
https://doi.org/10.18653/v1/N19-1423. Token sequence; masked-token prediction;
bidirectional; labels downstream. Not traffic/flow evidence.

### GPT

A. Radford *et al.*, “Improving Language Understanding by Generative
Pre-Training,” OpenAI, 2018. Primary URL:
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.
Autoregressive causal LM; not traffic evidence.

### CPC

A. van den Oord *et al.*, “Representation Learning with Contrastive Predictive
Coding,” arXiv:1807.03748, 2018. https://arxiv.org/abs/1807.03748. Future
prediction/contrastive, causal-capable; not flow-specific.

### SimCLR

T. Chen *et al.*, “A Simple Framework for Contrastive Learning of Visual
Representations,” ICML, 2020. https://proceedings.mlr.press/v119/chen20j.html.
Augmented-view contrastive; not traffic-specific.

### BYOL

J.-B. Grill *et al.*, “Bootstrap Your Own Latent: A New Approach to
Self-Supervised Learning,” *NeurIPS*, 2020.
https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html.
EMA teacher/student; no negatives; not traffic-specific.

### DINO

M. Caron *et al.*, “Emerging Properties in Self-Supervised Vision
Transformers,” ICCV, 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html.
EMA teacher/student; not traffic-specific.

### Barlow

J. Zbontar *et al.*, “Barlow Twins: Self-Supervised Learning via Redundancy
Reduction,” *ICML*, 2021.
https://proceedings.mlr.press/v139/zbontar21a.html. Cross-correlation
redundancy reduction; not traffic-specific.

### VICReg

A. Bardes *et al.*, “VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning,” *ICLR*, 2022.
https://openreview.net/forum?id=xm6YD62D1Ub.
Variance/invariance/covariance SSL; not traffic-specific.

### MAE

K. He *et al.*, “Masked Autoencoders Are Scalable Vision Learners,” CVPR,
2022. https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html.
Masked reconstruction; no traffic claim.

### data2vec

A. Baevski *et al.*, “data2vec: A General Framework for Self-Supervised
Learning in Speech, Vision and Language,” *ICML*, 2022.
https://proceedings.mlr.press/v162/baevski22a.html. A masked student predicts
contextual representations from a full-input EMA teacher, using an average of
the teacher’s top normalized transformer blocks. This is the direct root for
M1-L’s same-event EMA target, not a future-prediction objective. It covers
speech, vision, and language rather than NetFlow and provides no causal-serving
or external NetFlow-transfer evidence; project reproduction is
**UNVERIFIED**.

### IJEPA

A. Assran *et al.*, “Self-Supervised Learning from Images with a Joint-
Embedding Predictive Architecture,” CVPR, 2023.
https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html.
https://arxiv.org/abs/2301.08243. A context encoder predicts representations
of masked image target blocks from an EMA target encoder; the target blocks
come from the same image rather than a future stream. It is peer reviewed and
establishes generic masked joint-embedding prediction, not causal
next-event/NetFlow learning. Official code:
https://github.com/facebookresearch/ijepa. Its reported ImageNet scale and
vision results are not comparable to this project; reproduction is
**UNVERIFIED**.

### VJEPA

A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and
N. Ballas, “Revisiting Feature Prediction for Learning Visual Representations
from Video,” arXiv:2404.08471, 2024, https://arxiv.org/abs/2404.08471.
Preprint. V-JEPA predicts latent target features from video context with an
EMA target encoder. It provides feature-prediction precedent, not NetFlow,
strictly causal future-only serving, endpoint-disjoint transfer, or an IDS
result. The paper reports pretraining on two million videos and frozen-backbone
vision results. Official code: https://github.com/facebookresearch/jepa;
reproduction is **UNVERIFIED**.

### TSJEPA

S. Ennadir, S. Golkar, and L. Sarra, “Joint Embeddings Go Temporal,”
arXiv:2509.25449v1, 29 Sep. 2025, https://arxiv.org/abs/2509.25449.
Preprint; the arXiv record reports acceptance at the *Workshop on Time Series
in the Age of Large Models*, NeurIPS 2024, but that publication chronology is
**UNVERIFIED**. The method adapts JEPA-style representation learning to
univariate time-series classification and forecasting, not NetFlow. It predicts
uniformly masked time-series patches with a 128-wide, two-head Transformer
predictor and EMA target encoder; it does not define causal future-only
pretraining. The paper reports frozen evaluation on sensor and forecasting
datasets using an NVIDIA V100. Official code:
https://github.com/Sennadir/TS_JEPA; results are author-reported and
reproduction is **UNVERIFIED**.

### LeNEPA

A. Chemeris, M. Jin, and R. Balestriero, “LeNEPA: No-Augmentation Next-Latent
Prediction for Time-Series Representation Learning,” arXiv:2607.00958v1,
1 Jul. 2026, https://arxiv.org/abs/2607.00958. Preprint; the record reports
acceptance at the 12th MiLeTS Workshop at KDD 2026. Official code:
https://github.com/langotime/lenepa-milets-2026. LeNEPA trains causal
next-latent prediction without augmentation and uses SIGReg, not an EMA
teacher. It directly removes generic causal next-latent novelty, while its
NetFlow modality, endpoint-incident target, external-transfer split,
parameters, compute, and numerical reproduction are **UNVERIFIED**.

## Tabular, temporal, graph, and state-space roots

### VIME

J. Yoon, Y. Zhang, J. Jordon, and M. van der Schaar, “VIME: Extending the
Success of Self- and Semi-Supervised Learning to Tabular Domain,” *NeurIPS*,
2020.
https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html.
Tabular mask estimation and reconstruction are relevant controls; the data are
not network flows.

### SCARF

P. Bahri *et al.*, “SCARF: Self-Supervised Contrastive Learning Using Random
Feature Corruption,” *ICLR*, 2022.
https://openreview.net/forum?id=CuVqYB0kZci.
Tabular feature corruption contrastive SSL; not flows.

### FTTransformer

Y. Gorishniy *et al.*, “Revisiting Deep Learning Models for Tabular Data,”
NeurIPS, 2021. https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html. Supervised tabular baseline.

### STRATS

S. Tipirneni and C. Reddy, “Self-Supervised Transformer for Sparse and
Irregularly Sampled Multivariate Clinical Time-Series,” arXiv:2107.14293,
2021, https://arxiv.org/abs/2107.14293. Official code:
https://github.com/sindhura97/STraTS. Observation-triplet embeddings and an
auxiliary forecasting task motivate explicit missingness and time-gap handling;
the evidence is clinical, not network traffic.

### TS2Vec

Z. Yue *et al.*, “TS2Vec: Towards Universal Representation of Time Series,”
*AAAI*, 2022. https://ojs.aaai.org/index.php/AAAI/article/view/20881.
Hierarchical contrastive time-series representations; not flows.

### TimeMAE

M. Cheng, X. Tao, Z. Liu, Q. Liu, H. Zhang, R. Zhang, and E. Chen,
“TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked
Autoencoders,” *WSDM 2026*, https://doi.org/10.1145/3773966.3778007;
preprint: https://arxiv.org/abs/2303.00320. Subseries masking, decoupled
visible/masked encoders, codeword prediction, and momentum latent regression
are direct objective precedent, but bidirectional generic time series are not
causal NetFlow IDS.

### TiMAE

Z. Li, Z. Rao, L. Pan, P. Wang, and Z. Xu, “Ti-MAE: Self-Supervised Masked
Time Series Autoencoders,” arXiv:2301.08871, 2023,
https://arxiv.org/abs/2301.08871. Point-level time-series masking and
reconstruction support a masked temporal baseline, not traffic-specific or
cross-network evidence; archival acceptance and official code are
**UNVERIFIED**.

### SetTransformer

J. Lee *et al.*, “Set Transformer: A Framework for Attention-Based
Permutation-Invariant Neural Networks,” *ICML*, 2019.
https://proceedings.mlr.press/v97/lee19d.html. Permutation-invariant set attention; not traffic evidence.

### Graphormer

C. Ying *et al.*, “Do Transformers Really Perform Bad for Graph Representation?”
NeurIPS, 2021. https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html. Graph structural biases; not dynamic traffic.

### DGI

P. Veličković *et al.*, “Deep Graph Infomax,” ICLR, 2019.
https://openreview.net/forum?id=rklz9iAcKQ. Node/graph mutual-information SSL; not flow graphs.

### GraphMAE2

W. Hou *et al.*, “GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph
Learner,” *WWW*, 2024. https://dl.acm.org/doi/10.1145/3589334.3645451.
Masked graph autoencoding with remasking; not traffic evidence.

### TGN

E. Rossi *et al.*, “Temporal Graph Networks for Deep Learning on Dynamic
Graphs,” arXiv:2006.10637, 2020. https://arxiv.org/abs/2006.10637. Event-time
memory/message passing; causal if events are ordered; not IDS validation.

### Mamba

A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State
Spaces,” arXiv:2312.00752, 2023.
https://arxiv.org/abs/2312.00752. Selective state-space sequence model; no traffic claim.

### Chronos

A. F. Ansari *et al.*, “Chronos: Learning the Language of Time Series,”
arXiv:2403.07815v3, 2024, https://arxiv.org/abs/2403.07815. Official code and
checkpoints: https://github.com/amazon-science/chronos-forecasting. T5 models
tokenize scaled/quantized values for probabilistic forecasting across 42
datasets. This is adjacent time-series foundation-model evidence, not NetFlow
representation or IDS evidence; its generative quantization is excluded from
the core progression.

## Adaptation and anomaly roots

### DAPT

S. Gururangan *et al.*, “Don’t Stop Pretraining: Adapt Language Models to
Domains and Tasks,” *ACL*, 2020.
https://aclanthology.org/2020.acl-main.740/. Domain-adaptive pretraining;
not network traffic.

### LoRA

E. Hu *et al.*, “LoRA: Low-Rank Adaptation of Large Language Models,” *ICLR*,
2022. https://openreview.net/forum?id=nZeVKeeFYf9.
Low-rank adaptation; not an IDS method.

### DeepSVDD

L. Ruff *et al.*, “Deep One-Class Classification,” ICML, 2018.
https://proceedings.mlr.press/v80/ruff18a.html. One-class hypersphere objective;
labels/training split define validity.

### EnergyOOD

W. Liu *et al.*, “Energy-based Out-of-distribution Detection,” NeurIPS, 2020.
https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html. Energy score OOD; calibration/data shift still require evaluation.

## Evaluation roots

### NTFMReview2026

R. Pérez-Jove, C. R. Munteanu, J. Dorado, A. Pazos, and J. Vázquez-Naya,
“Network Traffic Foundation Models: A Systematic Review,” *Computer
Networks*, vol. 276, 111998, 2026,
https://doi.org/10.1016/j.comnet.2026.111998. Its review of 51 primary studies
finds limited cross-task transfer and scarce terabyte-scale corpora. It maps
the field but supplies no matched experiment for this proposal; the project’s
dated primary-source ledger and refreshed novelty search remain necessary.

### Dedup

K. Lee *et al.*, “Deduplicating Training Data Makes Language Models Better,”
*ACL 2022*, https://aclanthology.org/2022.acl-long.577/. This establishes that
duplicate training examples can distort memorization and evaluation in an
adjacent foundation-model domain. It does not define traffic equivalence;
exact and near-duplicate NetFlow keys must still be specified and audited
before split assignment.

### ComputeScaling

J. Kaplan *et al.*, “Scaling Laws for Neural Language Models,” arXiv:2001.08361,
2020, https://arxiv.org/abs/2001.08361. J. Hoffmann *et al.*, “Training
Compute-Optimal Large Language Models,” arXiv:2203.15556, 2022,
https://arxiv.org/abs/2203.15556. These motivate measured data/model scaling;
they do not prove network-traffic scaling. Report unique flows, exposures,
hardware, FLOPs, and transfer—not pretraining loss alone.

### NetBench

C. Qian, X. Li, Q. Wang, G. Zhou, and H. Shao, “NetBench: A Large-Scale and
Comprehensive Network Traffic Benchmark Dataset for Foundation Models,”
arXiv:2403.10319v2, 19 Mar. 2024,
https://arxiv.org/abs/2403.10319. It combines seven public traffic datasets
into 15 classification and five generation tasks. It is packet-traffic
oriented, and reuse of constituent benchmark traffic complicates independent
transfer; it is not a validated NetFlow deployment benchmark.

### ShortcutLearning

R. Geirhos *et al.*, “Shortcut Learning in Deep Neural Networks,” *Nature
Machine Intelligence*, 2020, doi: https://doi.org/10.1038/s42256-020-00257-z.
Primary evidence for shortcut risk; not an IDS experiment.

### PRvsROC

J. Davis and M. Goadrich, “The Relationship Between Precision-Recall and ROC
Curves,” ICML, 2006. https://dl.acm.org/doi/10.1145/1143844.1143874. Evaluate
rare alerts with PR as well as ROC. T. Saito and M. Rehmsmeier, “The
Precision-Recall Plot Is More Informative Than the ROC Plot When Evaluating
Binary Classifiers on Imbalanced Datasets,” *PLOS ONE*, vol. 10, no. 3,
e0118432, 2015, https://doi.org/10.1371/journal.pone.0118432. Report class
prevalence and operating-point alert volume; neither AUPRC nor AUROC alone is
an operational result.

### Calibration

C. Guo *et al.*, “On Calibration of Modern Neural Networks,” ICML, 2017.
https://proceedings.mlr.press/v70/guo17a.html. Report calibration on held-out
deployment-relevant data, not only accuracy.

### StatisticalTests

J. Demšar, “Statistical Comparisons of Classifiers over Multiple Data Sets,”
JMLR, 2006. https://jmlr.org/papers/v7/demsar06a.html. Use paired repeated
splits/datasets and disclose multiplicity; a single split is insufficient.

## Official standards and data sources

### IPFIXRFC7011

B. Claise *et al.*, “Specification of the IP Flow Information Export (IPFIX)
Protocol for the Exchange of Flow Information,” RFC 7011, 2013.
https://doi.org/10.17487/RFC7011. Official protocol standard.

### IPFIXRFC7012

B. Claise and B. Trammell, “Information Model for IP Flow Information Export
(IPFIX),” RFC 7012, 2013. https://doi.org/10.17487/RFC7012. Official
information-element registry/model.

### NetFlowV9RFC3954

Cisco, “Cisco Systems NetFlow Services Export Version 9,” RFC 3954, 2004.
https://doi.org/10.17487/RFC3954. Official NetFlow v9 format.

### NF3UQData

University of Queensland, “NetFlow Datasets,” official collection:
https://staff.itee.uq.edu.au/marius/NIDS_datasets/. This is the official release
for the four v3 datasets identified in [NF3Temporal](#nf3temporal). There is no
official merged `NF-UQ-NIDS-v3`; `NF-UQ-NIDS-v2` is a distinct legacy
collection. Project assumption: the supplied NF3 datasets are accepted
benchmark inputs. Their controlled, converted lineage supports benchmark
comparison, not claims about independent operational capture or deployment
transfer.

### CTU13

S. García, M. Grill, J. Stiborek, and A. Zunino, “An empirical comparison of
botnet detection methods,” *Computers & Security*, vol. 45, pp. 100–123, 2014,
https://doi.org/10.1016/j.cose.2014.05.011. Official dataset:
https://stratosphere-ips.squarespace.com/datasets-ctu13; authoritative scenario
files: https://mcfp.felk.cvut.cz/publicDatasets/datasets.html. Dataset. CTU-13
provides 13 separately captured scenarios with flow labels `Background`,
`Botnet`, `C&C Channels`, and `Normal`; scenarios—not arbitrary rows—are the
defensible split groups. The official description distinguishes `Background`
from `Normal`; excluding Background from clean-benign fitting is a conservative
project policy, not a property asserted by the dataset authors.
The public scenario material does not establish one versioned schema across all
scenarios, Argus version/timeouts, normalized timezone, a cryptographic release
record, or explicit redistribution terms. These limitations constrain
cross-scenario and operational claims; they do not make the traffic unusable as
a benchmark.

### CESNETTLSYear22

K. Hynek, J. Luxemburk, J. Pešek, T. Čejka, and P. Šiška,
“CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from backbone
lines,” *Scientific Data*, vol. 11, art. 1156, 2024,
https://doi.org/10.1038/s41597-024-03927-4. Official dataset/tooling:
https://cesnet.github.io/cesnet-datazoo/datasets_overview/ and
https://github.com/CESNET/cesnet-datazoo; versioned data record:
https://doi.org/10.5281/zenodo.10608607. Dataset. This 2022 corpus contains
507,739,073 sampled bidirectional TLS flows from five CESNET3 backbone probes,
with 180 service labels in 24 categories. The source records `ipfixprobe`
4.0.0--4.7.1, `ipfixcol2` 2.2.1, a five-minute active timeout, and a 65-second
idle timeout. Its Technical Validation trains on week T and tests on T+1
through T+8;
it recommends separate weeks 1–9 and 11–52 because the week-10 exporter update
changed packet-sequence distributions.
`TLS_SNI` supplies ground-truth service labels, while `TLS_SNI`, `APP`, and
`CATEGORY` are label-bearing fields and must not be model inputs for that
probe. Starts are clipped to the hour and ends are adjusted to preserve
duration; records do not identify the contributing probe. Those facts block
strict causal endpoint-history reconstruction. The official Zenodo API records
CC BY 4.0; the advertised schema discrepancy around source ports/PPI and the
selected dates limit comparisons.

### CICIot2022

S. Dadkhah, H. Mahdikhani, P. Kyei Danso, A. Zohourian, K. A. Truong, and
A. A. Ghorbani, “Towards the Development of a Realistic Multidimensional IoT
Profiling Dataset,” *2022 19th Annual International Conference on Privacy,
Security & Trust (PST)*, pp. 1–11, 2022,
https://doi.org/10.1109/PST55820.2022.9851966. Official dataset:
https://www.unb.ca/cic/datasets/iotdataset-2022.html. Dataset. The CIC page
describes power, idle, interaction, scenario, active, and attack captures,
including multi-device settings and repeated capture groups. It is a device
probe only if the acquired instance/device and capture-period group structure
is auditable; otherwise device-disjoint claims are **UNVERIFIED**. Raw MAC/IP
identifiers are not evidence of transferable device behaviour.

### UGR16

G. Maciá-Fernández *et al.*, “UGR’16: A New Dataset for the Evaluation of
Cybersecurity Intrusion Detection Systems,” *Computer Security*, 2018.
Primary DOI: https://doi.org/10.1016/j.cose.2018.03.004. Official dataset:
https://nesg.ugr.es/nesg-ugr16/. Real ISP background spans months, while test
attacks are injected; June contains documented anomaly risk and is not assumed
benign. The release exceeds 16.9 billion unidirectional NetFlow v9 records and
publishes weekly nfcapd/CSV files, but the official sources do not publish a
licence, timezone, per-record collector identity, or cryptographic release
record. These gaps limit strict chronology and collector-specific claims.

### LITNET2020

R. Damasevicius, A. Venckauskas, S. Grigaliunas, J. Toldinas,
N. Morkevicius, T. Aleliunas, and P. Smuikys, “LITNET-2020: An Annotated
Real-World Network Flow Dataset for Network Intrusion Detection,”
*Electronics*, vol. 9, no. 5, 800, 2020,
https://doi.org/10.3390/electronics9050800. Official data:
https://dataset.litnet.lt/data.php. The authors report 85 features and 12
attack types across 45,330,333 academic-network flows. Published material does
not resolve exporter version/timeouts, timezone, bidirectional completion,
formal redistribution terms, or release hashes. Downloaded schema, label rules,
chronology, and split suitability are not established by the published material.

### CICDocs

Canadian Institute for Cybersecurity, “Datasets,” official documentation.
https://www.unb.ca/cic/datasets/. Dataset-specific primary pages (e.g.,
CICDDoS2019) must supply collection/labels; benchmark splits are not implied.

### MAWI

WIDE Project, “MAWI Working Group Traffic Archive,” official archive.
https://mawi.wide.ad.jp/mawi/; privacy and use rules:
https://mawi.wide.ad.jp/mawi/guideline.txt; capture FAQ:
https://mawi.wide.ad.jp/mawi/faq.html. WIDE permits research use and prohibits
privacy-invasive use; redistribution of derived flow/pairing data is not
clearly granted. Samplepoint-F provides timestamped, anonymized backbone PCAP,
but endpoint mappings for ordinary daily traces are stable only within one
trace, timestamps have NTP/commodity-capture limitations, and mirrored traffic
may be incomplete or asymmetric. A frozen local PCAP-to-bidirectional-flow
conversion cannot establish stable cross-trace endpoint identity from the
published material. The traces may support exploration, unlabeled SSL, drift,
or alert-volume analysis, not IDS ground truth.
MAWILab is
a separate anomaly-label service for MAWI samplepoints B and F: R. Fontugne,
P. Borgnat, P. Abry, and K. Fukuda, “MAWILab: Combining diverse anomaly
detectors for automated anomaly labeling and performance benchmarking,” *CoNEXT
2010*, https://www.fukuda-lab.org/mawilab/. MAWILab data updates stopped in
December 2024; do not treat post-2024 MAWI traces as MAWILab-labelled data.

## Explicit exclusions

The requested “contextualised-NetFlow review” remains unresolved/excluded: no
stable identifier was located. The unresolved `GNNet` label is recorded above.
Unsourced summaries, marketing pages, and papers without adequate method
access are not used as novelty evidence; the labeled systematic review maps the
field but cannot establish a mechanism claim.
