# LANL Graph Autoencoder Baseline

## Purpose

This component implements a reconstruction-based graph anomaly baseline for the Aegis-HGX LANL homogeneous graph.

Unlike the earlier GCN, GraphSAGE, and GAT baselines, this model does not train directly against node anomaly labels. It learns node embeddings by reconstructing observed graph relationships.

## Files

* `src/aegis_hgx/models/baselines/train_lanl_graph_autoencoder.py`
* `configs/lanl_graph_autoencoder.yaml`
* `artifacts/models/lanl/lanl_graph_autoencoder.pt`
* `artifacts/models/lanl/lanl_graph_autoencoder_embeddings.pt`
* `reports/lanl_graph_autoencoder_metrics.json`

## Input Contract

The trainer expects a serialized PyTorch Geometric `Data` object containing:

* `x`: floating-point node features with shape `[N, F_in]`
* `edge_index`: integer graph connectivity with shape `[2, E]`

Node labels may be present but are not used by the reconstruction objective.

## Graph Preparation

The source LANL graph contains directed event relationships and repeated interactions between identical entity pairs.

For this baseline:

1. The graph object is cloned so the source artifact remains unchanged.
2. Stored self-loops are removed from reconstruction targets.
3. Duplicate directed pairs are collapsed.
4. Missing reverse edges are added.
5. The resulting graph is treated as an undirected structural graph.

This conversion is required because the baseline uses a symmetric inner-product decoder:

`score(i, j) = score(j, i)`

Direction, frequency, recency, and typed-relation semantics are intentionally deferred to later heterogeneous and temporal models.

## Edge Splitting

The model uses edge-level rather than node-level splitting:

* 70% training relationships
* 15% validation relationships
* 15% test relationships

`edge_index` contains message-passing connectivity visible to the encoder.

`edge_label_index` contains node pairs the decoder must score.

Validation-positive edges are excluded from validation message passing. Test-positive edges are excluded from test message passing. An explicit leakage audit verifies both conditions.

Training negatives are sampled dynamically on every epoch. Negative sampling excludes the complete known graph so validation and test positives cannot accidentally become training negatives.

Validation and test use fixed negative sets for stable metric comparison.

## Architecture

The encoder contains two graph-convolution layers:

`[N, F_in] → [N, 64] → [N, 32]`

The first layer is followed by ReLU and dropout.

The final layer has no ReLU so latent coordinates may be positive or negative.

The decoder calculates:

`probability(i, j) = sigmoid(z_iᵀ z_j)`

## Training Objective

Observed training relationships are positive examples.

Sampled non-relationships are negative examples.

The reconstruction loss encourages:

* positive-edge probabilities toward 1
* negative-edge probabilities toward 0

Node embeddings are not independent parameters. Gradients flow through edge probabilities and dot products into the GCN encoder parameters.

## Checkpoint Selection

The model trains for the configured number of epochs.

The checkpoint with the highest validation average precision is restored before final testing.

The final epoch is not automatically treated as the best model.

## Metrics

The trainer reports:

* reconstruction loss
* ROC-AUC
* average precision
* mean positive-edge probability
* mean negative-edge probability
* mean positive-edge anomaly score

These metrics measure held-out link reconstruction against sampled non-edges.

They do not directly measure malicious-event classification.

Because validation and test use a configured negative-sampling ratio, average precision reflects that artificial evaluation prevalence rather than real production edge prevalence.

## Anomaly Score

For an observed candidate relationship:

`anomaly_score(i, j) = 1 - reconstructed_probability(i, j)`

A high value means that the observed relationship is structurally incompatible with patterns learned by the encoder.

It is evidence of unusual topology, not proof of malicious activity.

A dedicated edge-level anomaly-scoring policy is implemented separately.

## Saved Embeddings

The embeddings artifact contains one latent vector per node with shape:

`[N, latent_channels]`

The saved embeddings are generated using the test message-passing graph. Training and validation connectivity may be available, but held-out test-positive relationships remain excluded.

## MLflow Tracking

Each run records:

* configuration parameters
* graph dimensions
* device
* per-epoch training loss
* per-epoch validation loss
* validation ROC-AUC
* validation average precision
* best checkpoint epoch
* final test metrics
* model checkpoint
* metrics report
* configuration
* embeddings artifact

## Limitations

1. The symmetric decoder cannot model edge direction.
2. Duplicate event frequency is discarded during structural coalescing.
3. Random edge splitting is not chronological leakage prevention.
4. Sampled non-edges are unobserved pairs, not guaranteed impossible relationships.
5. High-degree nodes may dominate reconstruction.
6. Link reconstruction quality does not automatically imply cyberattack-detection quality.
7. The full-batch GCN encoder may not scale to substantially larger graphs.
