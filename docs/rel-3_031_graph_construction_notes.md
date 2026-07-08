# Graph Construction Notes

## Purpose

This document defines how AEGIS-HGX will move from tabular cyber-event modeling into graph-structured cyber modeling.

The key design decision is:

```text
CIC remains the completed tabular baseline dataset.
LANL becomes the primary graph construction dataset.
```

The goal is to avoid forcing a weak graph structure onto a dataset artifact that was mainly prepared for tabular machine learning.

## Phase

This is graph data modeling and training/evaluation preparation.

It is not inference.

No serving behavior changes are introduced here.

## Why the Graph Strategy Changes

The completed CIC baseline laboratory produced a strong tabular evidence base:

* CIC ingestion
* clean tabular features
* logistic regression baseline
* XGBoost baseline
* scikit-learn MLP baseline
* PyTorch MLP diagnostic baseline
* calibration analysis
* seed-stability analysis
* baseline comparison report
* baseline laboratory release report

That work remains valuable.

However, graph modeling requires stable entities and relationships.

A cleaned tabular feature matrix is often designed to remove identity-like columns because those identifiers can cause leakage or overfitting in tabular models.

Graph construction has the opposite requirement.

A graph needs identity.

This creates the central distinction:

```text
tabular modeling wants clean numeric rows
graph modeling needs entities and relationships
```

## Core Graph Concepts

### Node

A node is an entity.

In a cyber graph, possible nodes include:

* user
* host
* source device
* destination device
* process
* IP address
* domain
* file
* port
* service

For the LANL graph phase, the most important initial node types are likely:

* user
* host
* device
* authentication source
* authentication destination

### Edge

An edge is a relationship between two nodes.

Examples:

```text
user -> logs_into -> host
host -> connects_to -> host
process -> connects_to -> destination
process -> writes -> file
```

For the first LANL graph, the most natural edge is an authentication or communication event between enterprise entities.

### Node Feature

A node feature describes an entity.

Examples:

* number of events
* number of unique neighbors
* inbound degree
* outbound degree
* failed authentication count
* successful authentication count
* rare-neighbor count
* activity frequency
* suspicious-edge ratio

### Edge Feature

An edge feature describes a relationship or event.

Examples:

* timestamp
* event type
* authentication success or failure
* source entity
* destination entity
* count
* time gap
* frequency
* protocol or service when available

### Label

A label defines the target used for training or evaluation.

Possible graph labels:

* edge label: suspicious relationship or event
* node label: suspicious user or host
* graph label: suspicious time window

For the first graph milestone, edge-level scoring is the cleanest direction because many cyber events naturally represent relationships.

## CIC Role

CIC remains the tabular benchmark.

Its role is to answer:

```text
What can strong non-graph tabular baselines already do?
```

CIC gives us the baseline bar for:

* PR-AUC
* ROC-AUC
* F1
* precision
* recall
* false positives
* false negatives
* calibration
* seed stability

CIC should not be discarded.

It should be used as the evidence base that future graph models must respect.

## LANL Role

LANL becomes the primary graph dataset.

Its role is to answer:

```text
Can entity relationships improve anomaly detection beyond tabular evidence?
```

LANL is a better graph substrate because enterprise telemetry naturally supports entity modeling.

The graph phase should focus on:

* users
* hosts
* authentication relationships
* host-to-host activity
* repeated access patterns
* rare relationships
* temporal behavior
* relationship-level risk

## First Graph Target

The first graph target is delibertaley simple:

```text
static enterprise entity graph
```

Initial interpretation:

```text
node = user or host
edge = authentication or communication event
edge features = event attributes
node features = aggregated behavior
edge label or node label = suspiciousness target where available
```

This should come before heterogeneous graph modeling and temporal graph modeling.

## Homogeneous Graph

A homogeneous graph has one broad node type and one broad edge type.

Example:

```text
entity -> interacts_with -> entity
```

Benefits:

* simpler to build
* easier to inspect
* easier to visualize
* easier to convert to a PyTorch Geometric `Data` object
* better first baseline target

Limitation:

* loses some semantic detail about entity and relationship types

## Heterogeneous Graph

A heterogeneous graph has multiple node and edge types.

Example:

```text
user -> logs_into -> host
host -> communicates_with -> host
process -> connects_to -> external_ip
process -> writes_file -> file
```

Benefits:

* more realistic
* preserves typed cyber relationships
* aligns better with future temporal edge-risk memory

Cost:

* more complex schema
* more complex feature engineering
* requires PyTorch Geometric `HeteroData`
* harder debugging and evaluation

AEGIS-HGX should begin with a simple homogeneous graph and then expand toward heterogeneous modeling.

## Static Graph

A static graph summarizes relationships across a selected dataset slice.

It does not explicitly model event order.

This is the correct first graph step because it lets us validate:

* entity extraction
* node IDs
* edge construction
* feature aggregation
* graph sanity checks
* connected components
* degree distributions

## Temporal Graph

A temporal graph preserves event order.

It supports questions such as:

* Did this relationship become risky over time?
* Did this user-host pattern drift?
* Did this process-to-destination pattern become unusual?
* How quickly can the model detect a low-and-slow attack?

Temporal modeling comes later, after static graph construction is correct.

## Proposed LANL Graph Tables

The next implementation phase should produce normalized event tables first, then graph tables.

### Event Table

Example columns:

```text
event_id
timestamp
event_type
source_entity
destination_entity
source_type
destination_type
event_result
label
```

### Node Table

Example columns:

```text
node_id
node_key
node_type
event_count
in_degree
out_degree
unique_neighbor_count
failed_event_count
successful_event_count
rare_neighbor_count
```

### Edge Table

Example columns:

```text
edge_id
source_node_id
destination_node_id
edge_type
timestamp
event_count
event_result
time_gap_seconds
label
```

## Future PyTorch Geometric Mapping

For a homogeneous graph, PyTorch Geometric will eventually need:

```text
x = node feature matrix
edge_index = source and destination node indices
edge_attr = edge feature matrix
y = labels
```

Conceptual mapping:

```text
node table -> x
edge table source/destination columns -> edge_index
edge table numeric features -> edge_attr
edge or node labels -> y
```

For heterogeneous modeling later, the project will use typed node stores and typed edge stores through `HeteroData`.

## Evaluation Principle

A graph model must not be considered better just because it is more complex.

It must improve meaningful outcomes such as:

* higher PR-AUC
* better false-positive behavior
* better false-negative behavior
* better relationship-level explanation
* better seed stability
* better calibration
* better low-and-slow detection later

The baseline laboratory exists to prevent graph-model hype.

## Next Implementation Direction

The next implementation milestone is LANL ingestion.

The intended sequence is:

1. ingest a LANL sample
2. normalize LANL event tables
3. build LANL node and edge tables
4. inspect graph structure with NetworkX
5. convert graph tables into PyTorch Geometric format
6. train first graph baselines
7. compare graph evidence against the tabular baseline bar