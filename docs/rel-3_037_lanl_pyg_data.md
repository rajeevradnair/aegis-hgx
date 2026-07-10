# LANL PyG Data Builder

## Purpose

This document explains the LANL PyTorch Geometric data-building workflow.

This step converts LANL graph node and edge tables into a homogeneous PyG `Data` object.

This is training/test data preparation. It is not inference and it is not model training.

## Inputs

The script reads:

```text
data/processed/lanl/graph_nodes.parquet
data/processed/lanl/graph_edges.parquet
```

These files are produced by the LANL graph table builder and inspected by the NetworkX graph inspection workflow.

## Main Script

```text
src/aegis_hgx/models/baselines/build_lanl_pyg_data.py
```

## Config

```text
configs/lanl_pyg_data.yaml
```

The config controls input paths, output paths, selected node features, selected edge features, and validation behavior.

## Outputs

The script writes:

```text
data/processed/lanl/lanl_homogeneous_graph.pt
data/processed/lanl/lanl_homogeneous_graph_metadata.json
```

The `.pt` file contains the serialized PyG `Data` object.

The metadata JSON records tensor shapes, feature columns, mappings, label values, input/output paths, and validation settings.

## PyG Data Fields

The PyG object contains:

```text
x
edge_index
edge_attr
y
node_id
node_type_id
edge_label
edge_type_id
event_family_id
edge_id
```

## Field Meanings

| Field             | Meaning                                    |
| ----------------- | ------------------------------------------ |
| `x`               | Node feature matrix                        |
| `edge_index`      | Graph connectivity tensor                  |
| `edge_attr`       | Edge feature matrix                        |
| `y`               | Node label tensor                          |
| `node_id`         | Original graph node IDs                    |
| `node_type_id`    | Encoded node entity types                  |
| `edge_label`      | Edge labels kept separate from model input |
| `edge_type_id`    | Encoded original edge types                |
| `event_family_id` | Encoded event families                     |
| `edge_id`         | Original edge IDs for traceability         |

## Node Features

Node features are built from:

```text
event_count
first_seen_timestamp
last_seen_timestamp
active_span
entity_type
```

`active_span` is computed as:

```text
last_seen_timestamp - first_seen_timestamp
```

Numeric node features are min-max scaled.

Categorical node features are one-hot encoded.

The node label is not included in `x`.

## Edge Features

Edge features are built from:

```text
timestamp
edge_type
```

Numeric edge features are min-max scaled.

Categorical edge features are one-hot encoded.

The edge label is not included in `edge_attr`.

## Leakage Prevention

The script avoids putting labels into model input tensors.

Node labels are stored in:

```text
y
```

Edge labels are stored separately in:

```text
edge_label
```

Red-team ground-truth edge type information is not allowed to become an edge input feature. The leakage placeholder column is dropped from `edge_attr` feature columns if present:

```text
edge_type_ground_truth_edge_type_withheld
```

This prevents the model from learning a shortcut from ground-truth metadata.

## Compact Node Indexing

PyG requires node indices in `edge_index` to be compact:

```text
0, 1, 2, ..., num_nodes - 1
```

The script creates:

```text
node_id -> pyg_node_index
```

In the current graph, `node_id` may already equal `pyg_node_index`, but the explicit mapping is still required for robustness.

This protects future workflows involving filtering, subgraphs, train/test masks, sampling, or batching.

## Invalid Edge Handling

Before building `edge_index`, the script checks every edge endpoint.

An edge is valid only if both IDs exist in the node table:

```text
source_node_id
destination_node_id
```

Invalid edges are excluded from PyG conversion.

Current policy:

```text
warn
```

Meaning:

```text
Invalid edges are counted, reported, and excluded.
```

For stricter validation, this can later be changed to:

```text
fail
```

## Validation

The script validates:

```text
required node schema
required edge schema
compact PyG node indices
edge_index shape
edge_index index range
x shape
y shape
edge_attr shape
edge_label shape
PyG Data object validity
```

The PyG object must pass:

```text
data.validate(raise_on_error=True)
```

## Run Command

```bash
python -m aegis_hgx.models.baselines.build_lanl_pyg_data --config configs/lanl_pyg_data.yaml
```

## Verify Saved Artifact

```bash
python - <<'PY'
import torch

data = torch.load(
    "data/processed/lanl/lanl_homogeneous_graph.pt",
    weights_only=False,
)

data.validate(raise_on_error=True)

print(data)
print("x:", data.x.shape)
print("edge_index:", data.edge_index.shape)
print("edge_attr:", data.edge_attr.shape)
print("y:", data.y.shape)
print("edge_label:", data.edge_label.shape)
PY
```

## Smoke Test

```bash
python -m pytest tests/test_build_lanl_pyg_data.py -ra
```

## Healthy Signals

A healthy PyG conversion should show:

```text
num_nodes > 0
num_edges > 0
x shape is [num_nodes, num_node_features]
edge_index shape is [2, num_edges]
edge_attr shape is [num_edges, num_edge_features]
y shape is [num_nodes]
edge_label shape is [num_edges]
invalid_edge_rows is 0 or very small
node_id_equals_pyg_index is true for the current full graph
PyG validation passes
```

## Why This Step Matters

Graph neural networks do not train directly on pandas tables.

They need tensor representations of graph structure and features.

This workflow converts the inspected LANL graph into a reusable PyG artifact that future GCN, GraphSAGE, GAT, graph autoencoder, and temporal graph models can load.
