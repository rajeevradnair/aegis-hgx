Heterogeneous GraphSAGE Baseline
Purpose

This baseline predicts whether LANL user and host nodes participated in confirmed red-team activity using their ordinary typed telemetry neighborhoods.

Process and host/domain nodes provide graph context but are not supervised prediction targets.LANL

Input Data

The graph is constructed from:

data/processed/lanl/graph_nodes.parquet
data/processed/lanl/graph_edges.parquet

Node types:

user
host
process
host_or_domain

Example relations:

user → auth_logon → host
user → process_start → process
host → dns_resolution → host_or_domain
host → network_flow → host
Leakage Prevention

All explicit red-team ground-truth edges are removed before feature construction and graph message passing.

Filtering excludes records derived from:

redteam_ground_truth
redteam_activity
confirmed_redteam
redteam.txt.gz
edge label 1

The node security labels remain as supervised targets.

Features

Each node receives seven numerical features calculated only from ordinary telemetry:

outgoing event count
incoming event count
unique-neighbor count
total event count
activity duration
normalized first-seen timestamp
normalized last-seen timestamp

Count and duration features use log1p scaling.

Graph Construction

Global node IDs are converted into node-type-local indexes because every PyG heterogeneous node store has its own local index space.

Repeated events between the same typed node pair are collapsed into one structural relation.

ToUndirected(merge=False) creates explicit reverse relation types so users, hosts, processes, and domains can receive messages from connected entities.

Model

The model uses two heterogeneous GraphSAGE layers.

Every canonical relation receives its own SAGEConv.

HeteroConv runs each relation-specific operator and sums messages arriving at the same destination node type.

Separate linear heads produce:

one logit per user
one logit per host
Training

Training uses class-weighted binary cross-entropy on user and host training masks.

The user and host losses are averaged.

The model checkpoint with the highest validation average precision is retained.

The classification threshold is selected using validation F1.

Test

The test masks are evaluated only after training, checkpoint selection, and threshold selection are complete.

Reported metrics include:

average precision
ROC-AUC
precision
recall
F1
alert count
Limitations

This is a static, transductive node-classification baseline.

The full graph structure is visible while node labels are divided into train, validation, and test masks. It is therefore not a chronological or unseen-node evaluation.

The model does not yet use:

temporal windows
event-level edge features
neighbor sampling
probability calibration
multi-seed confidence intervals
production alert-volume policies