# LANL Graph Inspection

## Purpose

This document explains the LANL graph inspection workflow used before converting graph tables into PyTorch Geometric objects and before training graph neural network models.

This step is graph validation and evaluation preparation. It is not model training and it is not inference.

## Inputs

The inspection script reads:

```text
data/processed/lanl/graph_nodes.parquet
data/processed/lanl/graph_edges.parquet
```

These files are produced by the LANL graph table builder.

## Main Script

```text
visualization/lanl_graph_viz.py
```

The script builds an in-memory NetworkX `MultiDiGraph` from the graph node and edge tables.

A `MultiDiGraph` is used because LANL events can contain repeated directed relationships between the same source and destination nodes. For example, the same user may authenticate to the same host many times at different timestamps.

## Config

```text
configs/lanl_graph_inspection.yaml
```

The config controls input paths, output paths, graph type, and inspection limits.

Important inspection settings:

```yaml
invalid_edge_policy: warn
invalid_edge_sample_size: 25
top_k_nodes: 25
component_sample_size: 10
redteam_sample_size: 25
```

## Invalid Edge Handling

Before building the NetworkX graph, the script checks whether every edge endpoint exists in `graph_nodes.parquet`.

An edge is valid only if both of these IDs exist in the node table:

```text
source_node_id
destination_node_id
```

Invalid edges are not added to the NetworkX graph.

This prevents NetworkX from silently creating ghost nodes that were not present in the original graph node table.

Current policy:

```text
warn
```

Meaning:

```text
Invalid edges are excluded, counted, and reported.
```

For stricter validation, the policy can later be changed to:

```text
fail
```

## Graph Checks Performed

The script computes the following graph sanity checks.

### Basic Counts

The script reports:

```text
node_count
edge_count
valid_edge_rows
invalid_edge_rows
unique_node_ids
unique_edge_ids
```

These checks confirm that the graph loaded correctly and that invalid edges were excluded.

### Node Distributions

The script reports:

```text
node_type_counts
node_label_counts
```

These checks confirm that expected entity types survived graph construction, such as:

```text
user
host
process
host_or_domain
```

### Edge Distributions

The script reports:

```text
edge_type_counts
event_family_counts
edge_label_counts
```

These checks confirm that event semantics survived graph construction, such as authentication, DNS, network flow, process activity, and red-team labels.

### Degree Statistics

The script computes:

```text
total_degree
in_degree
out_degree
```

These identify high-activity entities.

Interpretation:

```text
total_degree = total graph activity touching a node
in_degree = how often a node is reached as a destination
out_degree = how often a node initiates activity as a source
```

This helps detect important hubs and possible data-quality issues, such as an accidental `unknown::unknown` mega-node.

### Connected Components

The script computes weakly and strongly connected component summaries.

Weak components ignore edge direction and answer:

```text
Are these entities connected at all?
```

Strong components respect edge direction and answer:

```text
Can activity flow both ways through directed paths?
```

The most important early signal is:

```text
largest_weak_component_ratio
```

A very low value may mean the graph is too fragmented for useful message passing.

### Isolated Nodes

The script counts nodes with no edges.

Because the node table is created from event endpoints, isolated nodes should usually be zero or very small.

Isolated nodes may indicate stale nodes, filtered edges, or graph table mismatch.

### Self-Loops

The script counts edges where:

```text
source_node_id == destination_node_id
```

Self-loops may be valid in some graphs, but they require inspection because they can indicate entity-resolution issues.

### Red-Team Profile

The script checks whether labeled red-team activity survived graph construction.

It reports:

```text
redteam_edge_count
redteam_participating_node_count
redteam_labeled_node_count
redteam_edge_type_counts
redteam_event_family_counts
redteam_node_type_counts
redteam_edge_sample
```

This confirms that known suspicious activity is visible in the graph before graph model training begins.

## Outputs

The script writes:

```text
reports/lanl_graph_profile.md
reports/lanl_graph_profile.json
reports/lanl_top_nodes.csv
```

### Markdown Report

```text
reports/lanl_graph_profile.md
```

Human-readable graph inspection report.

### JSON Report

```text
reports/lanl_graph_profile.json
```

Machine-readable graph profile for future automation, checks, or release reports.

### Top Nodes CSV

```text
reports/lanl_top_nodes.csv
```

CSV export of the highest-degree nodes by:

```text
total_degree
in_degree
out_degree
```

This file is useful for manual graph review and failure analysis.

## Run Command

```bash
python visualization/lanl_graph_viz.py --config configs/lanl_graph_inspection.yaml
```

## Expected Healthy Signals

A healthy graph inspection should show:

```text
invalid_edge_rows = 0 or very small
node_count > 0
edge_count > 0
expected entity types present
expected edge types present
red-team labels present if redteam data exists
isolated_node_count = 0 or very small
largest_weak_component_ratio not extremely low
```

## Why This Step Matters

Graph neural networks depend heavily on graph structure.

If the graph is malformed, fragmented, mislabeled, or full of ghost nodes, later GCN, GraphSAGE, GAT, and temporal graph models may train successfully but produce misleading results.

This inspection step protects the modeling phase by validating graph structure before tensor conversion and graph model training.
