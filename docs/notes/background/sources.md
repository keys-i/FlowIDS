# Other sources

Standards, dataset pages, software links, and project history are kept here.
These are not research-paper reviews. The entries come from the earlier
reference list; current availability was not rechecked.

## HistoricalPrototype828582d

keys-i, “added script to get all datasets,” commit `828582d0a2b6a8f8fbf090dae84328490563629f`, 21 May 2026 UTC, https://github.com/keys-i/FlowIDS/commit/828582d0a2b6a8f8fbf090dae84328490563629f.

History of the old eight-flow, bidirectional prototype. No fixed split,
checkpoint, or result was recorded.

## GNNet

The earlier list did not identify a paper, DOI, or repository for
`GNNet graph foundation model`. We need that before planning a comparison.

## CESNETModels

CESNET, “cesnet-models,” recorded repository link:
https://github.com/CESNET/cesnet-models.

Possible packet teachers for X1. Check availability, compatible inputs, and
exact paired records before use.

## IPFIXRFC7011

B. Claise *et al.*, RFC 7011, 2013, https://doi.org/10.17487/RFC7011.

Defines the IPFIX protocol.

## IPFIXRFC7012

B. Claise and B. Trammell, RFC 7012, 2013, https://doi.org/10.17487/RFC7012.

Defines the IPFIX information model.

## NetFlowV9RFC3954

Cisco, RFC 3954, 2004, https://doi.org/10.17487/RFC3954.

Defines the NetFlow v9 format.

## NF3UQData

University of Queensland, “NetFlow Datasets,” https://staff.itee.uq.edu.au/marius/NIDS_datasets/.

Current runs use only [NF-CSE-CIC-IDS2018-v3](https://doi.org/10.48610/ECE9B83).
The [Hugging Face mirror](https://huggingface.co/datasets/keys-i/netFlow) names
its file `data/NF-CICIDS2018-v3.parquet`; locally it is saved as
`data/NF-CSE-CIC-IDS2018-v3.parquet`. This is a converted benchmark; results
still need checking on independent operational traffic.

## CICDocs

Canadian Institute for Cybersecurity, “Datasets,” https://www.unb.ca/cic/datasets/.

Documents how each dataset was collected and labelled.

## MAWI

WIDE Project, “MAWI Working Group Traffic Archive,” https://mawi.wide.ad.jp/mawi/.

Possible unlabelled training and drift data. The earlier notes found no
stable identity across traces or IDS ground truth, and recorded that
MAWILab stopped updates in December 2024. Check the reuse rules before use.

## NetFlow_Transformer

Read from `tmp/NetFlow_Transformer` on 11 September 2026: `main`,
`IP_to_IP_branch`, and `Jepa_Model`. No reference code was run.

The main branch builds the embedding, encoder, and prediction head explicitly.
The IP-to-IP branch changes input features and time ordering. The JEPA branch
separates context encoder, EMA target encoder, predictor, training, and diagnostics.
Its target is a future flow; our M1 teacher predicts hidden fields at the same flow.

The useful pattern is a short entrypoint and components named for their jobs.
Our command now enters `src/main.py` directly, and both training paths share
optimizer setup. The JEPA branch fits scaling and category maps on its training
split. Our run likewise keeps training-only preprocessing, with a purge and
separate validation and test periods. It now reads/splits once and shares M1
feature tensors between pretraining and classification.
