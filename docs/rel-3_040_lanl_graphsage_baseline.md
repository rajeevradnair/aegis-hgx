# LANL GraphSAGE Baseline

## Purpose

This document explains the LANL GraphSAGE baseline.

The goal is to train an inductive graph neural network baseline on the LANL homogeneous PyTorch Geometric graph.

GraphSAGE follows the earlier GCN baseline. GCN showed normalized message passing. GraphSAGE extends the graph modeling path by focusing on neighbor aggregation and inductive learning.

## Main Files

```text
configs/lanl_graphsage_pyg.yaml
src/aegis_hgx/models/baselines/train_lanl_graphsage_pyg.py
tests/test_train_lanl_graphsage_pyg.py
```

## Output Artifacts

```text
artifacts/models/lanl/lanl_graphsage_model.pt
reports/lanl_graphsage_metrics.json
```

## Input Artifact

```text
artifacts/models/lanl/lanl_homogeneous_graph.pt
```

This file contains the PyG `Data` object built from the LANL graph tables.

## Core Idea

GraphSAGE means:

```text
Graph SAmple and aggreGatE
```

The key idea is:

```text
A node representation should depend on:
1. the node's own features
2. a summary of its neighbors' features
```

For a cyber graph, this matters because risk is relational.

A host may look normal by itself, but suspicious when connected to unusual users, rare processes, or risky domains.

## GCN vs GraphSAGE

| Topic               | GCN                                         | GraphSAGE                                 |
| ------------------- | ------------------------------------------- | ----------------------------------------- |
| Core idea           | Normalize and mix neighbors using adjacency | Aggregate neighbor features               |
| Typical aggregation | normalized sum                              | mean, max, or LSTM aggregation            |
| New unseen nodes    | less naturally inductive                    | designed for inductive use                |
| Scaling idea        | often full-graph message passing            | neighbor sampling and aggregation         |
| Cyber intuition     | good static graph baseline                  | better fit for evolving enterprise graphs |

## Cybersecurity Example

Example graph:

```text
U17 logs_into H42
P9 runs_on H42
P9 connects_to D5
```

Where:

```text
U17 = user
H42 = host
P9  = process
D5  = domain
```

A tabular model may only see isolated features.

GraphSAGE lets `H42` learn from:

```text
H42's own host features
+
mean summary of connected users, processes, and domains
```

This helps the model capture relational risk.

## PyG Data Fields Used

| Field             |                       Shape | Meaning                      |
| ----------------- | --------------------------: | ---------------------------- |
| `data.x`          | `[num_nodes, num_features]` | node feature matrix          |
| `data.edge_index` |            `[2, num_edges]` | graph connectivity           |
| `data.y`          |               `[num_nodes]` | node labels                  |
| `data.train_mask` |               `[num_nodes]` | nodes used for training loss |
| `data.val_mask`   |               `[num_nodes]` | nodes used for validation    |
| `data.test_mask`  |               `[num_nodes]` | nodes used for final testing |

## GraphSAGE Model

The model is a two-layer GraphSAGE network:

```text
data.x + data.edge_index
    -> SAGEConv
    -> ReLU
    -> Dropout
    -> SAGEConv
    -> logits
```

The first `SAGEConv` layer creates hidden node embeddings.

The second `SAGEConv` layer creates raw class scores.

## Model Class

```text
LanlGraphSAGE
```

Architecture:

```text
conv1: SAGEConv(input_channels, hidden_channels, aggr="mean")
conv2: SAGEConv(hidden_channels, output_channels, aggr="mean")
```

For binary node classification:

```text
output_channels = 2
```

The output shape is:

```text
[num_nodes, 2]
```

Each row contains:

```text
[class_0_score, class_1_score]
```

## Aggregation

This baseline uses:

```text
aggregation = mean
```

Mean aggregation summarizes neighbor information by averaging neighbor representations.

Plain English:

```text
For each node, look at its neighbors, average their features, then combine that summary with the node's own features.
```

## Training Phase

Training uses only `train_mask` nodes for supervised loss.

Flow:

```text
model.train()
optimizer.zero_grad()
logits = model(data.x, data.edge_index)
train_loss = cross_entropy(logits[train_mask], y[train_mask])
train_loss.backward()
optimizer.step()
```

Important detail:

```text
The forward pass runs over the full graph.
The loss is calculated only on training nodes.
```

This is because message passing needs graph connectivity, but supervised training labels should only come from the training split.

## Validation Phase

Validation uses `val_mask` nodes.

Flow:

```text
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    val_loss = cross_entropy(logits[val_mask], y[val_mask])
```

Validation loss is used to choose the best checkpoint.

The final epoch is not automatically the best model.

## Test Phase

Test evaluation uses the best validation checkpoint.

Flow:

```text
load best model state
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    evaluate logits[test_mask]
```

Metrics saved:

```text
accuracy
precision
recall
F1
ROC-AUC
PR-AUC
test positive label count
test total node count
```

## Inference Phase

Production-style inference would use:

```text
model.eval()
with torch.no_grad():
    logits = model(x, edge_index)
    probabilities = softmax(logits)
    suspicious_probability = probabilities[:, positive_label]
```

GraphSAGE is useful for inference on evolving cyber graphs because new users, hosts, processes, or domains can be scored using their features and neighborhoods.

## Class Imbalance Handling

Cyber anomaly labels are often imbalanced.

The trainer computes class weights from the training labels:

```text
rare classes receive higher weight
common classes receive lower weight
```

This helps suspicious nodes matter more during cross-entropy loss.

## Saved Checkpoint

The checkpoint stores:

```text
model_state_dict
model_class
input_channels
hidden_channels
output_channels
dropout
aggregation
config
saved_at_utc
```

This allows the GraphSAGE model to be reconstructed later.

## Saved Metrics JSON

The metrics file stores:

```text
run metadata
graph path
model path
metrics path
model configuration
training configuration
split counts
best validation epoch
test metrics
training history
```

This makes the experiment auditable and comparable against the GCN baseline.

## Run Command

```bash
python -m aegis_hgx.models.baselines.train_lanl_graphsage_pyg --config configs/lanl_graphsage_pyg.yaml
```

## Test Command

```bash
python -m pytest tests/test_train_lanl_graphsage_pyg.py -ra
```

## Inspect Metrics

```bash
python -m json.tool reports/lanl_graphsage_metrics.json | head -120
```

## Inspect Checkpoint

```bash
python - <<'PY'
import torch

checkpoint = torch.load(
    "artifacts/models/lanl/lanl_graphsage_model.pt",
    weights_only=False,
)

print(checkpoint.keys())
print(checkpoint["model_class"])
print(checkpoint["input_channels"])
print(checkpoint["hidden_channels"])
print(checkpoint["output_channels"])
print(checkpoint["aggregation"])
PY
```

## Healthy Signals

A healthy run should show:

```text
LANL PyG graph summary for GraphSAGE training
Node mask summary
GraphSAGE model summary
GraphSAGE forward-pass smoke check summary
Training GraphSAGE model
GraphSAGE training summary
GraphSAGE test-set metrics
Saved GraphSAGE training outputs
```

Healthy tensor expectations:

```text
data.x          -> [num_nodes, num_features]
data.edge_index -> [2, num_edges]
data.y          -> [num_nodes]
logits          -> [num_nodes, num_classes]
```

## Common Failure Modes

| Failure                               | Meaning                              | Fix                                             |
| ------------------------------------- | ------------------------------------ | ----------------------------------------------- |
| `data.x` missing                      | graph artifact lacks node features   | rerun PyG graph builder                         |
| `edge_index` wrong shape              | graph connectivity malformed         | expect `[2, num_edges]`                         |
| logits shape wrong                    | model output mismatch                | expect `[num_nodes, num_classes]`               |
| no positives in test split            | rare-label split issue               | inspect mask summary                            |
| high accuracy but low recall          | class imbalance                      | focus on recall and PR-AUC                      |
| train improves but validation worsens | overfitting                          | reduce epochs, hidden size, or increase dropout |
| GraphSAGE underperforms GCN           | features or aggregation may not help | compare carefully, do failure analysis          |

## Interview Explanation

GraphSAGE is an inductive graph neural network method.

Instead of learning an embedding table tied to fixed node IDs, it learns how to generate a node embedding by aggregating information from the node's neighborhood.

In AEGIS-HGX, this matters because enterprise cyber graphs evolve continuously. New users, hosts, processes, and domains appear over time. GraphSAGE is a better fit for this setting because it can score nodes from features and neighborhood structure rather than relying only on previously seen node identities.

## Key Takeaways

```text
GraphSAGE = sample and aggregate.

A node embedding depends on the node's own features and its neighbors' features.

Mean aggregation is the simplest GraphSAGE aggregator.

SAGEConv uses data.x and data.edge_index.

Training loss is calculated only on train_mask nodes.

Validation loss selects the best checkpoint.

Test metrics are computed only after training is complete.

GraphSAGE is more naturally inductive than GCN.

For cyber anomaly detection, GraphSAGE helps model relational risk across users, hosts, processes, and domains.
```
