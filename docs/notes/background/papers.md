# Earlier paper notes

These entries came from the August reading list. Their source details and
claims were not rechecked in this revision. The papers checked in September
have [separate notes](../README.md).

## EGraphSAGE

W. W. Lo *et al.*, “E-GraphSAGE,” *NOMS*, 2022, https://doi.org/10.1109/NOMS54207.2022.9789878. Code: https://github.com/waimorris/E-GraphSAGE.

Classifies graph edges using labels. A comparison for graph context, although
our model learns from a sequence of past flows.

## AnomalE

E. Caville *et al.*, “Anomal-E,” *Knowledge-Based Systems*, 2022, https://doi.org/10.1016/j.knosys.2022.110030. Code: https://github.com/waimorris/Anomal-E.

Learns graph representations without labels. It does not test Transformer
pretraining on unlabelled NetFlow.

## NFStandardFeatures

M. Sarhan *et al.*, “NetFlow Datasets for Machine Learning-Based NIDS,” 2020, https://doi.org/10.1007/978-3-030-72802-1_9; “Towards a Standard Feature Set,” 2022, https://doi.org/10.1007/s11036-021-01843-0.

Explains where the shared flow features come from. These papers do not test
pretraining or establish a chronological split.

## NF3Temporal

M. Luay *et al.*, “Temporal Analysis of NetFlow Datasets,” arXiv:2503.04404, 2025; “Time Matters,” *IEEE Access*, 2026, https://doi.org/10.1109/ACCESS.2026.3688204.

Describes timestamps and the four v3 datasets. We still need to define and
check our own split.

## FlowSequenceBERT

L. G. Nguyen and K. Watabe, “Flow Sequence and BERT,” *IEEE ICC*, 2023, https://doi.org/10.1109/ICC45041.2023.10279335.

Uses flow sequences and domain adaptation. The earlier review found no usable
code or weights.

## NetFlowGen

J. Zhou *et al.*, “NetFlowGen,” arXiv:2412.20635, 2024, https://arxiv.org/abs/2412.20635.

A generative approach to NetFlow pretraining. The earlier review found no
official code or checkpoint.

## NEGSC

R. Xu *et al.*, “Applying SSL to NIDS for Network Flows with GNN,” *Computer Networks*, 2024, https://doi.org/10.1016/j.comnet.2024.110356. Code: https://github.com/renj-xu/NEGSC.

Learns from graph structure. Compare with our time-ordered endpoint histories.

## GraphIDS

L. Guerra *et al.*, “Self-Supervised Learning of Graph Representations for NID,” *NeurIPS*, 2025, https://arxiv.org/abs/2509.16625. Code: https://github.com/lorenzo9uerra/GraphIDS.

Combines masked traffic with graph context and scores reconstruction errors.
That leaves our past-only, cross-network transfer question open.

## VanLangendonckGraphFM

L. Van Langendonck *et al.*, “Towards a Graph-Based Foundation Model,” 2024, https://arxiv.org/abs/2409.08111.

Pretrains on dynamic graphs. The earlier review found no released large checkpoint.

## TimestampGraphContrastive

J. Dai *et al.*, “Timestamp-Aware Spatio-Temporal Graph Contrastive Learning,” arXiv:2606.17109, 2026, https://arxiv.org/abs/2606.17109. Code: https://github.com/Rory6235/STG-NIDS.

A time-aware graph comparison. Its reported setting does not establish our
past-only input rule.

## TSDSGNN

H. Bai *et al.*, “SSL IDS Based on Dynamic Spatiotemporal Graph,” *Applied Intelligence*, 2026, https://doi.org/10.1007/s10489-026-07371-2.

The earlier review could not access enough method detail to plan a comparison.

## TrafficMAE

W. Zhu *et al.*, “TrafficMAE,” *ICT Express*, 2025, https://doi.org/10.1016/j.icte.2025.11.004.

Masked reconstruction on packets and sessions. The earlier review found no
usable code or weights; applying it to NetFlow would need an adaptation.

## netFound

S. Beltiukov *et al.*, “netFound,” arXiv:2310.17025, revised 2026, https://arxiv.org/abs/2310.17025. Code: https://github.com/SNL-UCSB/netFound; weights: https://huggingface.co/snlucsb.

Uses hierarchy in large-scale traffic pretraining. Relevant to M5, but its
tokens and tasks differ from ours.

## Lens

X. Li *et al.*, “Lens,” arXiv:2402.03646, https://arxiv.org/abs/2402.03646.

Masks spans in packet/flow context. The earlier review did not verify reusable
code, weights, or independent transfer results.

## NetGPT

X. Meng *et al.*, “NetGPT,” arXiv:2304.09513, https://arxiv.org/abs/2304.09513.

Generates packet/flow sequences causally. The earlier review found no usable
code or weights.

## TrafficGPT

J. Qu *et al.*, “TrafficGPT,” arXiv:2403.05822, 2024, https://arxiv.org/abs/2403.05822. Checkpoint: https://huggingface.co/LaBackDoor/trafficgpt.

Pretrains on packet traffic. Benchmark results leave independent IDS transfer untested.

## ETBERT

X. Lin *et al.*, “ET-BERT,” *WWW*, 2022, https://doi.org/10.1145/3485447.3512217. Code and checkpoint: https://github.com/linwhitehat/ET-BERT.

Pretrains for encrypted packet/burst classification. The recorded code and
checkpoint make it a possible packet-model comparison.

## YaTC

R. Zhao *et al.*, “Yet Another Traffic Classifier,” *AAAI*, 2023, https://doi.org/10.1609/aaai.v37i4.25674. Code and checkpoint: https://github.com/NSSL-SJTU/YaTC.

A possible packet teacher for X1. First check that its preprocessing works
with our paired captures.

## NetMamba

T. Wang *et al.*, “NetMamba,” *ICNP*, 2024, https://arxiv.org/abs/2405.11449. Code: https://github.com/wangtz19/NetMamba.

Useful for comparing efficiency. It uses a different model type and does not
establish detection of unseen attacks.

## TrafficMoE

J. Zhou *et al.*, “Traffic-MoE,” arXiv:2601.00357, 2026, https://arxiv.org/abs/2601.00357.

The earlier review found no official code. Sparse routing is outside the current plan.

## FlowletFormer

L. Liu *et al.*, “FlowletFormer,” arXiv:2508.19924, 2025, https://arxiv.org/abs/2508.19924.

Uses flowlets and field context. Relevant to history design, but the earlier
review found no usable code or weights.

## MM4flow

L. Yang *et al.*, “MM4flow,” *ACM CCS*, 2025, https://doi.org/10.1145/3719027.3744804.

Uses packet and flow inputs at scale. The earlier review found no released
code or weights.

## NetVAD

D. Fürst *et al.*, “NetVAD,” arXiv:2606.01452, 2026, https://arxiv.org/abs/2606.01452.

Trains on benign traffic without identifiers. Its evaluation still uses labels
for calibration.

## Rosetta

R. Xie *et al.*, “Rosetta,” *USENIX Security*, 2023, https://www.usenix.org/conference/usenixsecurity23/presentation/xie. Code: https://github.com/sunskyXX/Rosetta.

Studies TLS augmentation. Applying it to general NetFlow pretraining would need more work.

## NetSSM

A. Chu *et al.*, “NetSSM,” *ACM on Networking*, 2026, https://doi.org/10.1145/3786289.

Generates traffic traces; it does not evaluate intrusion detection.

## DenoisingAutoencoder

P. Vincent *et al.*, “Denoising Autoencoders,” *ICML*, 2008, https://doi.org/10.1145/1390156.1390294.

Background for learning to reconstruct damaged inputs.

## BERT

J. Devlin *et al.*, “BERT,” *NAACL-HLT*, 2019, https://doi.org/10.18653/v1/N19-1423.

Background for masked-input learning. Its experiments concern language.

## GPT

A. Radford *et al.*, “Improving Language Understanding by Generative Pre-Training,” 2018, https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

Background for causal pretraining. Its experiments concern language.

## CPC

A. van den Oord *et al.*, “Contrastive Predictive Coding,” arXiv:1807.03748, 2018, https://arxiv.org/abs/1807.03748.

Background for learning representations by predicting later inputs.

## SimCLR

T. Chen *et al.*, “SimCLR,” *ICML*, 2020, https://proceedings.mlr.press/v119/chen20j.html.

Background for contrastive learning on images.

## BYOL

J.-B. Grill *et al.*, “BYOL,” *NeurIPS*, 2020, https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html.

Background for the slowly updated teacher used in M1 teacher.

## DINO

M. Caron *et al.*, “DINO,” *ICCV*, 2021, https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html.

Background for self-supervised learning on images.

## Barlow

J. Zbontar *et al.*, “Barlow Twins,” *ICML*, 2021, https://proceedings.mlr.press/v139/zbontar21a.html.

A comparison for reducing repeated information across vector dimensions.

## VICReg

A. Bardes *et al.*, “VICReg,” *ICLR*, 2022, https://openreview.net/forum?id=xm6YD62D1Ub.

Uses variance and covariance to prevent constant representations.

## MAE

K. He *et al.*, “Masked Autoencoders,” *CVPR*, 2022, https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html.

Background for masked reconstruction.

## TalkLikeAPacket

S. Mayhoub *et al.*, “Talk Like a Packet: Rethinking Network Traffic Analysis
with Transformer Foundation Models,” arXiv:2602.06636, 2026,
https://arxiv.org/abs/2602.06636.

An overview of packet-language pretraining. The earlier review found no model
artifact linked from the arXiv record; it added no experiment to the plan.

## VIME

J. Yoon *et al.*, “VIME,” *NeurIPS*, 2020, https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html.

A masking comparison for tabular inputs.

## SCARF

P. Bahri *et al.*, “SCARF,” *ICLR*, 2022, https://openreview.net/forum?id=CuVqYB0kZci.

A contrastive comparison using corrupted features.

## FTTransformer

Y. Gorishniy *et al.*, “Revisiting Deep Learning Models for Tabular Data,” *NeurIPS*, 2021, https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html.

A supervised comparison for tabular data.

## STRATS

S. Tipirneni and C. Reddy, “STraTS,” arXiv:2107.14293, 2021, https://arxiv.org/abs/2107.14293.

Background for representing missing values and gaps between events.

## TS2Vec

Z. Yue *et al.*, “TS2Vec,” *AAAI*, 2022, https://ojs.aaai.org/index.php/AAAI/article/view/20881.

A comparison for hierarchy in time-series learning.

## TimeMAE

M. Cheng *et al.*, “TimeMAE,” *WSDM*, 2026, https://doi.org/10.1145/3773966.3778007.

Background for separating masking and teacher-target prediction.

## TiMAE

Z. Li *et al.*, “Ti-MAE,” arXiv:2301.08871, 2023, https://arxiv.org/abs/2301.08871.

A masked-reconstruction comparison for time series.

## SetTransformer

J. Lee *et al.*, “Set Transformer,” *ICML*, 2019, https://proceedings.mlr.press/v97/lee19d.html.

Background for attention over sets.

## Graphormer

C. Ying *et al.*, “Graphormer,” *NeurIPS*, 2021, https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html.

Background for adding graph structure to attention scores.

## DGI

P. Veličković *et al.*, “Deep Graph Infomax,” *ICLR*, 2019, https://openreview.net/forum?id=rklz9iAcKQ.

A comparison for self-supervised graph learning.

## GraphMAE2

W. Hou *et al.*, “GraphMAE2,” *WWW*, 2024, https://dl.acm.org/doi/10.1145/3589334.3645451.

Background for masked reconstruction on graphs.

## TGN

E. Rossi *et al.*, “Temporal Graph Networks,” arXiv:2006.10637, 2020, https://arxiv.org/abs/2006.10637.

A comparison for keeping memory of earlier graph events.

## Mamba

A. Gu and T. Dao, “Mamba,” arXiv:2312.00752, 2023, https://arxiv.org/abs/2312.00752.

A state-space model for comparing runtime and memory use.

## Chronos

A. F. Ansari *et al.*, “Chronos,” arXiv:2403.07815, 2024, https://arxiv.org/abs/2403.07815. Code: https://github.com/amazon-science/chronos-forecasting.

A forecasting comparison; intrusion detection would need a separate test.

## DAPT

S. Gururangan *et al.*, “Don’t Stop Pretraining,” *ACL*, 2020, https://aclanthology.org/2020.acl-main.740/.

Background for adapting a pretrained model to another domain.

## LoRA

E. Hu *et al.*, “LoRA,” *ICLR*, 2022, https://openreview.net/forum?id=nZeVKeeFYf9.

A comparison for adapting a model with fewer trainable parameters.

## DeepSVDD

L. Ruff *et al.*, “Deep One-Class Classification,” *ICML*, 2018, https://proceedings.mlr.press/v80/ruff18a.html.

A one-class detection comparison.

## EnergyOOD

W. Liu *et al.*, “Energy-based OOD Detection,” *NeurIPS*, 2020, https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html.

A comparison for detecting inputs outside the training distribution.

## ConMD

X. Lian *et al.*, “Contextual Masking Distillation for Network Traffic Anomaly
Detection,” *IEEE Transactions on Information Forensics and Security*, 2026,
https://doi.org/10.1109/TIFS.2026.3655514. Code:
https://github.com/ikun0124/ConMD.

Packet and flow distillation is related to X1. I still need exact paired data and compatible
preprocessing.

## NTFMReview2026

R. Pérez-Jove *et al.*, “Network Traffic Foundation Models: A Systematic Review,” *Computer Networks*, 2026, https://doi.org/10.1016/j.comnet.2026.111998.

An overview of the field. Check the original methods before claiming something is new.

## Dedup

K. Lee *et al.*, “Deduplicating Training Data,” *ACL*, 2022, https://aclanthology.org/2022.acl-long.577/.

Background for keeping duplicate training and test records apart.

## ComputeScaling

J. Kaplan *et al.*, arXiv:2001.08361, 2020, https://arxiv.org/abs/2001.08361; J. Hoffmann *et al.*, arXiv:2203.15556, 2022, https://arxiv.org/abs/2203.15556.

Measure data, model, FLOPs, and transfer, not pretraining loss alone.

## NetBench

C. Qian *et al.*, “NetBench,” arXiv:2403.10319, 2024, https://arxiv.org/abs/2403.10319.

A packet benchmark. It does not test deployment on independent NetFlow traffic.

## ShortcutLearning

R. Geirhos *et al.*, “Shortcut Learning,” *Nature Machine Intelligence*, 2020, https://doi.org/10.1038/s42256-020-00257-z.

Background for checking whether models use accidental clues in the data.

## PRvsROC

J. Davis and M. Goadrich, *ICML*, 2006, https://dl.acm.org/doi/10.1145/1143844.1143874; T. Saito and M. Rehmsmeier, *PLOS ONE*, 2015, https://doi.org/10.1371/journal.pone.0118432.

Report PR, ROC, prevalence, and alert volume.

## Calibration

C. Guo *et al.*, “On Calibration,” *ICML*, 2017, https://proceedings.mlr.press/v70/guo17a.html.

Calibrate on held-out relevant data.

## StatisticalTests

J. Demšar, “Statistical Comparisons of Classifiers,” *JMLR*, 2006, https://jmlr.org/papers/v7/demsar06a.html.

Background for statistical comparisons across repeated experiments.

## CTU13

S. García *et al.*, “An Empirical Comparison of Botnet Detection Methods,” *Computers & Security*, 2014, https://doi.org/10.1016/j.cose.2014.05.011. Data: https://stratosphere-ips.squarespace.com/datasets-ctu13.

A possible test with held-out scenarios. Fields and release details differ
across scenarios.

## CESNETTLSYear22

K. Hynek *et al.*, “CESNET-TLS-Year22,” *Scientific Data*, 2024, https://doi.org/10.1038/s41597-024-03927-4. Data: https://github.com/CESNET/cesnet-datazoo.

A possible test over a longer span of TLS traffic. Keep label fields out of
inputs. Missing probe identity prevents reliable endpoint histories.

## CICIot2022

S. Dadkhah *et al.*, “CIC IoT Dataset 2022,” *PST*, 2022, https://doi.org/10.1109/PST55820.2022.9851966. Data: https://www.unb.ca/cic/datasets/iotdataset-2022.html.

Check capture groups before claiming that training and test devices are separate.

## UGR16

G. Maciá-Fernández *et al.*, “UGR’16,” *Computers & Security*, 2018, https://doi.org/10.1016/j.cose.2018.03.004. Data: https://nesg.ugr.es/nesg-ugr16/.

Injected attacks and gaps in metadata limit the chronological tests we can claim.

## LITNET2020

R. Damasevicius *et al.*, “LITNET-2020,” *Electronics*, 2020, https://doi.org/10.3390/electronics9050800. Data: https://dataset.litnet.lt/data.php.

Check flow order and exporter boundaries before use.

## XION

[Paper link](https://arxiv.org/abs/2608.26831). The earlier notes excluded it
because its dependency-based anomaly score changed none of the planned
pretraining comparisons. It was not reviewed again here.
