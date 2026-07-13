# From-Scratch GCN Mechanics

## Purpose

This document explains the from-scratch Graph Convolutional Network implementation.

The goal is to understand what a GCN layer does mechanically before relying on PyTorch Geometric's `GCNConv`.

This is a learning and training-mechanics lab. It is not production inference.

## Main File

```text
src/aegis_hgx/research/from_scratch_gcn.py
```

## Test File

```text
tests/test_from_scratch_gcn.py
```

## Core Question

A normal neural network layer transforms each row independently.

```text
node_features -> linear layer -> new node_features
```

A GCN layer transforms each node using both its own features and its neighbors' features.

```text
node_features + graph_structure -> graph convolution -> new node_features
```

## Core Formula

The simplified GCN layer is:

```text
H_next = A_norm @ H @ W
```

Where:

| Symbol   | Meaning                                                |
| -------- | ------------------------------------------------------ |
| `H`      | Current node feature matrix or hidden embedding matrix |
| `A_norm` | Normalized adjacency matrix                            |
| `W`      | Learnable weight matrix                                |
| `H_next` | Updated node representations                           |

## Plain-English Intuition

A GCN layer does two things:

```text
1. Mix each node with its neighbors.
2. Learn a transformation of the mixed information.
```

In code:

```python
mixed_neighbor_features = normalized_adjacency @ x
output = mixed_neighbor_features @ weight
```

## Toy Graph

The toy graph has 4 nodes:

```text
0 = user-like node
1 = host-like node
2 = process-like node
3 = domain-like node
```

The graph structure is:

```text
0 -- 1
1 -- 2
1 -- 3
```

The adjacency matrix is:

```text
[
  [0, 1, 0, 0],
  [1, 0, 1, 1],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
]
```

`A[i, j] = 1` means node `i` is connected to node `j`.

## Node Features

The toy node feature matrix is:

```text
x shape = [4, 3]
```

Meaning:

```text
4 nodes
3 features per node
```

Each row is one node.

Each column is one feature.

## Labels

The toy labels are:

```text
y = [0, 1, 0, 1]
```

Meaning:

```text
node 0 -> class 0
node 1 -> class 1
node 2 -> class 0
node 3 -> class 1
```

For the cyber framing:

```text
0 = benign
1 = suspicious
```

## Why Add Self-Loops?

A raw adjacency matrix only lets a node receive information from its neighbors.

If we add self-loops, each node also keeps its own information.

```text
A_self = A + I
```

This means:

```text
new node representation = own information + neighbor information
```

Without self-loops, a node could lose its own identity after message passing.

## Degree

The degree of a node is the number of connections it has.

After adding self-loops, degree includes the node's connection to itself.

Example:

```text
node 1 connects to node 0, node 2, node 3, and itself
degree = 4
```

## Why Normalize Adjacency?

Without normalization, high-degree nodes can dominate message passing.

A node connected to many neighbors would receive a much larger summed signal than a low-degree node.

Normalization keeps the scale controlled.

The implementation uses symmetric GCN normalization:

```text
A_norm = D^(-1/2) @ A_self @ D^(-1/2)
```

Where:

| Symbol     | Meaning                           |
| ---------- | --------------------------------- |
| `A_self`   | Adjacency matrix with self-loops  |
| `D`        | Degree matrix                     |
| `D^(-1/2)` | Inverse square-root degree matrix |
| `A_norm`   | Normalized adjacency matrix       |

## Manual GCN Layer

The manual layer is implemented as:

```python
class ManualGCNLayer(nn.Module):
    ...
```

It owns one learnable parameter:

```text
weight
```

The weight shape is:

```text
[input_features, output_features]
```

The forward pass does:

```python
mixed_neighbor_features = normalized_adjacency @ x
output = mixed_neighbor_features @ self.weight
```

Shape flow:

```text
normalized_adjacency: [num_nodes, num_nodes]
x:                    [num_nodes, input_features]
weight:               [input_features, output_features]

normalized_adjacency @ x:
[num_nodes, num_nodes] @ [num_nodes, input_features]
= [num_nodes, input_features]

mixed_neighbor_features @ weight:
[num_nodes, input_features] @ [input_features, output_features]
= [num_nodes, output_features]
```

## Tiny Two-Layer Manual GCN

The two-layer model is:

```python
class TinyManualGCN(nn.Module):
    ...
```

Architecture:

```text
x
  -> ManualGCNLayer
  -> ReLU
  -> ManualGCNLayer
  -> logits
```

The first layer creates hidden node representations.

The second layer creates class logits.

For the toy graph:

```text
input x shape = [4, 3]
hidden shape = [4, 4]
logits shape = [4, 2]
```

## Why ReLU Matters

Without ReLU, two stacked linear GCN layers would still behave like one larger linear transformation.

ReLU adds nonlinearity.

That lets the model learn more complex patterns.

## Logits

The model output is called `logits`.

Logits are raw class scores.

For 4 nodes and 2 classes:

```text
logits shape = [4, 2]
```

Example:

```text
node 0 -> [class 0 score, class 1 score]
node 1 -> [class 0 score, class 1 score]
node 2 -> [class 0 score, class 1 score]
node 3 -> [class 0 score, class 1 score]
```

## Cross-Entropy Loss

Training uses cross-entropy loss:

```python
loss = F.cross_entropy(logits, y)
```

Important:

```text
Pass raw logits into cross_entropy.
Do not apply softmax before cross_entropy.
```

PyTorch's `F.cross_entropy` internally applies log-softmax.

## Backward Pass

The key training line is:

```python
loss.backward()
```

This computes gradients for:

```text
model.layer1.weight
model.layer2.weight
```

Gradient flow:

```text
loss
<- logits
<- second manual GCN layer
<- ReLU
<- first manual GCN layer
<- normalized adjacency multiplication
<- node features
```

## Optimizer Step

After gradients are computed, the optimizer updates the model weights:

```python
optimizer.step()
```

This changes:

```text
model.layer1.weight
model.layer2.weight
```

## Gradient Inspection

The implementation inspects gradient norms:

```text
layer1_gradient_norm
layer2_gradient_norm
```

If both are greater than zero, then both layers received learning signal.

## Weight Update Inspection

The implementation compares initial weights to final weights.

It reports:

```text
layer1_weight_change_norm
layer2_weight_change_norm
```

If both are greater than zero, then both layers were actually updated.

## Training Phase

During training:

```text
model.train()
optimizer.zero_grad()
logits = model(x, normalized_adjacency)
loss = F.cross_entropy(logits, y)
loss.backward()
optimizer.step()
```

This is supervised node classification on a tiny toy graph.

## Inference Phase

Inference would be:

```text
model.eval()
with torch.no_grad():
    logits = model(x, normalized_adjacency)
    predictions = logits.argmax(dim=1)
```

The current lab focuses on training mechanics, not production inference.

## What the Test Proves

The test verifies:

```text
normalized adjacency has the expected shape
normalized adjacency has finite values
self-loop diagonal values are positive
manual GCN layer produces the expected hidden shape
two-layer manual GCN produces logits
cross-entropy loss works
loss.backward() creates gradients in both layers
optimizer.step() changes both layers' weights
```

## Run Command

```bash
python -m aegis_hgx.research.from_scratch_gcn
```

## Test Command

```bash
python -m pytest tests/test_from_scratch_gcn.py -ra
```

## Healthy Signals

A healthy run should show:

```text
normalized_adjacency_shape: [4, 4]
hidden_output_shape: [4, 2]
logits_shape: [4, 2]
loss decreases over epochs
layer1_gradient_norm > 0
layer2_gradient_norm > 0
layer1_weight_change_norm > 0
layer2_weight_change_norm > 0
```

## Key Takeaways

```text
A graph is represented by node features and adjacency.

A GCN layer mixes neighbor features using normalized adjacency.

Self-loops let nodes keep their own information.

Degree normalization prevents high-degree nodes from dominating.

The learnable weight matrix transforms mixed neighbor information.

ReLU adds nonlinearity between GCN layers.

Cross-entropy trains the model against node labels.

loss.backward() proves gradients flow through the manual GCN.

optimizer.step() proves weights actually update.

This from-scratch implementation explains what PyG's GCNConv abstracts away.
```
