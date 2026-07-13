# LANL GCN Baseline

## Purpose

This document explains the first LANL Graph Convolutional Network baseline.

The goal of this baseline is to train a simple supervised graph neural network on the LANL homogeneous PyTorch Geometric graph artifact.

This is training and evaluation work. It is not production inference.

## Input Artifact

The trainer loads the PyG graph artifact created by the LANL PyG data builder:

```text
artifacts/models/lanl/lanl_homogeneous_graph.pt
```

The graph is a homogeneous PyG `Data` object.

## Main Script

```text
src/aegis_hgx/models/baselines/train_lanl_gcn_pyg.py
```

## Config

```text
configs/lanl_gcn_pyg.yaml
```

The config controls:

```text
input graph path
output model path
output metrics path
train/validation/test split ratios
random seed
hidden dimension
dropout
epochs
learning rate
weight decay
positive label
```

## Outputs

The trainer writes:

```text
artifacts/models/lanl/lanl_gcn_model.pt
reports/lanl_gcn_metrics.json
```

The model checkpoint stores the best validation-loss model weights.

The metrics JSON stores training history, split counts, best validation epoch, and test metrics.

## PyG Fields Used by the First GCN

The first GCN baseline uses:

| Field             | Used For                        |
| ----------------- | ------------------------------- |
| `data.x`          | Node features                   |
| `data.edge_index` | Graph connectivity              |
| `data.y`          | Node labels                     |
| `data.train_mask` | Nodes used for training loss    |
| `data.val_mask`   | Nodes used for validation loss  |
| `data.test_mask`  | Nodes used for final evaluation |

The first GCN baseline does not use:

```text
data.edge_attr
data.edge_label
data.edge_type_id
data.event_family_id
```

Those fields are still useful for later edge-aware, heterogeneous, temporal, and explainability models.

## Model Architecture

The baseline uses a 2-layer GCN:

```text
node features
    -> GCNConv
    -> ReLU
    -> Dropout
    -> GCNConv
    -> node-class logits
```

The first layer maps raw node features into hidden node embeddings.

The second layer maps hidden embeddings into class logits.

For binary node classification, the output shape is:

```text
[num_nodes, 2]
```

Each node receives two raw scores:

```text
class 0 score
class 1 score
```

## Training Objective

The model is trained with cross-entropy loss.

Training loss is calculated only on nodes selected by:

```text
data.train_mask
```

Validation loss is calculated only on nodes selected by:

```text
data.val_mask
```

Test metrics are calculated only on nodes selected by:

```text
data.test_mask
```

## Why Node Masks Are Used

This is a single-graph node-classification setup.

The graph structure remains shared, but labels are split by masks.

```text
same graph
different node subsets
different training/evaluation roles
```

The masks prevent the training loss from being computed on validation or test nodes.

## Class Imbalance Handling

Cyber anomaly labels are usually imbalanced.

The trainer builds class weights from the training labels only.

Rare classes receive larger loss weight.

This helps the model avoid learning the trivial behavior:

```text
predict everything as benign
```

## Training Flow

The training loop follows this sequence:

```text
1. Load config.
2. Load PyG graph.
3. Validate graph fields.
4. Create train/validation/test node masks.
5. Build the 2-layer GCN model.
6. Run a forward-pass smoke check.
7. Build class weights.
8. Build optimizer.
9. Train for multiple epochs.
10. Track validation loss.
11. Save the best validation-loss checkpoint.
12. Evaluate on the test mask.
13. Save model checkpoint and metrics JSON.
```

## Forward Pass

The GCN forward pass receives:

```text
data.x
data.edge_index
```

and returns:

```text
logits
```

where:

```text
logits shape = [num_nodes, num_classes]
```

For the LANL graph, this means every node receives a prediction.

## Training Phase

During training:

```text
model.train()
dropout is active
gradients are tracked
loss is computed on train_mask nodes
optimizer updates model weights
```

## Validation Phase

During validation:

```text
model.eval()
dropout is disabled
gradients are not tracked
loss is computed on val_mask nodes
weights are not updated
```

Validation loss is used to choose the best checkpoint.

## Test Evaluation Phase

During test evaluation:

```text
best validation-loss checkpoint is loaded
model.eval()
dropout is disabled
gradients are not tracked
metrics are computed on test_mask nodes
```

The test set is used only after training is complete.

## Metrics

The trainer reports:

| Metric    | Meaning                                                           |
| --------- | ----------------------------------------------------------------- |
| Accuracy  | Fraction of correctly classified test nodes                       |
| Precision | Of predicted suspicious nodes, how many were truly suspicious     |
| Recall    | Of truly suspicious nodes, how many were caught                   |
| F1        | Balance between precision and recall                              |
| ROC-AUC   | Ranking quality across thresholds                                 |
| PR-AUC    | Precision-recall quality, especially important for rare positives |

For cyber anomaly detection, accuracy alone is not enough.

PR-AUC, recall, precision, and F1 are more important because suspicious nodes are rare.

## Why This Baseline Matters

This GCN baseline is the first graph neural model in the project.

It establishes a working graph learning pipeline:

```text
LANL graph artifact
    -> PyG Data object
    -> GCN model
    -> supervised node classification
    -> metrics
    -> saved checkpoint
```

This baseline becomes the comparison point for later models:

```text
GraphSAGE
GAT
graph autoencoder
heterogeneous GNN
temporal GNN
Temporal Edge-Risk Memory
```

## Current Limitations

This first baseline is intentionally simple.

Known limitations:

```text
It uses a homogeneous graph.
It does not use edge_attr.
It does not use edge labels.
It does not use temporal ordering.
It does not use relation-specific message passing.
It uses random/stratified node masks rather than chronological temporal splits.
It is not yet production inference.
```

These limitations are acceptable for the first graph baseline.

Later phases will address them with heterogeneous graphs, temporal splits, temporal memory, edge-risk scoring, ablations, and robustness testing.

## Run Command

```bash
python -m aegis_hgx.models.baselines.train_lanl_gcn_pyg --config configs/lanl_gcn_pyg.yaml
```

## Smoke Test

```bash
python -m pytest tests/test_train_lanl_gcn_pyg.py -ra
```

## Inspect Metrics

```bash
python -m json.tool reports/lanl_gcn_metrics.json | head -120
```

## Inspect Checkpoint

```bash
python - <<'PY'
import torch

checkpoint = torch.load(
    "artifacts/models/lanl/lanl_gcn_model.pt",
    weights_only=False,
)

print(checkpoint.keys())
print(checkpoint["model_class"])
print(checkpoint["input_channels"])
print(checkpoint["hidden_channels"])
print(checkpoint["output_channels"])
PY
```

## Healthy Signals

A healthy first GCN run should show:

```text
graph loads successfully
node masks are created
forward-pass smoke check passes
training loss is finite
validation loss is finite
test metrics are written
model checkpoint is saved
metrics JSON is saved
smoke test passes
```

## Cheat Sheet

```text
GCN = graph neural network that updates each node using neighbor information.

data.x = node features.

data.edge_index = graph structure.

data.y = node labels.

train_mask = nodes used for training loss.

val_mask = nodes used to choose the best checkpoint.

test_mask = nodes used for final evaluation.

GCNConv does message passing over edges.

Cross entropy compares logits against class labels.

Dropout is active during training and disabled during evaluation.

Best validation-loss checkpoint is used for test evaluation.

Accuracy can be misleading for rare cyber attacks.

PR-AUC, recall, precision, and F1 matter more for anomaly detection.
```
