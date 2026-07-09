# LANL Graph Data Pipeline

## Purpose

This document describes the LANL data pipeline that converts raw LANL telemetry into graph-ready node and edge tables for downstream graph anomaly detection research.

The pipeline supports future NetworkX inspection, PyTorch Geometric graph construction, GCN, GraphSAGE, GAT, heterogeneous graph models, temporal graph models, and Temporal Edge-Risk Memory experiments.

## Phase Classification

This pipeline is offline data preparation for future training and evaluation.

It is not inference.

No serving API, prediction endpoint, online feature lookup, or production scoring path is changed by this workflow.

## Pipeline Narrative

```text
raw LANL files
        ↓
processed LANL parquet files
        ↓
clean LANL event tables
        ↓
graph node and edge tables
        ↓
NetworkX inspection
        ↓
PyTorch Geometric graph object
        ↓
static graph models
        ↓
temporal graph models
```

## Pipeline Stages

### Stage 1: LANL Ingestion

Script:

```text
pipelines/ingest_lanl_sample.py
```

Config:

```text
configs/lanl_ingest.yaml
```

Input directory:

```text
data/external/lanl
```

Expected input files:

```text
auth.txt.gz
dns.txt.gz
flows.txt.gz
proc.txt.gz
redteam.txt.gz
```

Output directory:

```text
data/processed/lanl
```

Expected outputs:

```text
auth.parquet
dns.parquet
flows.parquet
proc.parquet
redteam.parquet
ingest_manifest.json
```

Purpose:

The ingestion stage reads compressed LANL source files, assigns canonical column names per event family, adds lineage metadata, and writes one parquet file per source event family.

Design decision:

Each source file is kept separate because authentication, DNS, network flow, process, and red-team records have different semantics.

### Stage 2: Clean Event Tables

Script:

```text
pipelines/build_lanl_event_tables.py
```

Config:

```text
configs/lanl_event_tables.yaml
```

Primary input:

```text
data/processed/lanl/auth.parquet
data/processed/lanl/dns.parquet
data/processed/lanl/flows.parquet
data/processed/lanl/proc.parquet
data/processed/lanl/redteam.parquet
```

Expected outputs:

```text
clean_auth_events.parquet
clean_dns_events.parquet
clean_flow_events.parquet
clean_process_events.parquet
clean_redteam_events.parquet
clean_all_events.parquet
event_table_manifest.json
```

Purpose:

The event-table stage maps different LANL source schemas into a shared canonical event schema.

Common event columns:

```text
event_id
timestamp
event_family
event_type
source_entity
destination_entity
source_entity_type
destination_entity_type
event_result
label
source_file
row_number
```

Design decisions:

Authentication events are represented as user-to-host events.

DNS events are represented as host-to-host-or-domain events.

Network-flow events are represented as host-to-host events.

Process events are represented as user-to-process events while preserving host context.

Red-team records are preserved as confirmed red-team rows with label 1.

Non-red-team rows use label 0.

Red-team labels are not aggressively joined onto other event families yet. More careful timestamp and entity matching can be added later.

### Stage 3: Graph Node and Edge Tables

Script:

```text
pipelines/build_lanl_graph_tables.py
```

Config:

```text
configs/lanl_graph_tables.yaml
```

Primary input:

```text
data/processed/lanl/clean_all_events.parquet
```

Expected outputs:

```text
graph_nodes.parquet
graph_edges.parquet
graph_table_manifest.json
```

Purpose:

The graph-table stage turns normalized events into graph-ready nodes and edges.

Graph nodes represent entities.

Graph edges represent relationships or events between entities.

Node identity rule:

```text
entity_type + node_key_separator + entity_name
```

Example node keys:

```text
user::U001
host::C001
process::powershell.exe
host_or_domain::corp.example
```

Node columns:

```text
node_id
entity_type
entity_name
node_key
first_seen_timestamp
last_seen_timestamp
event_count
label
```

Edge columns:

```text
edge_id
source_node_id
destination_node_id
source_entity
destination_entity
source_entity_type
destination_entity_type
edge_type
event_family
event_type
timestamp
event_result
label
source_file
row_number
```

Design decisions:

Each clean event becomes one graph edge.

Each unique entity becomes one graph node.

Node IDs are stable because nodes are sorted by entity type and entity name before IDs are assigned.

Edges inherit labels from clean events.

Nodes receive label 1 if they participate in any labeled red-team edge.

## Why Graph Tables Matter

Graph models need numeric node IDs and structured edge relationships.

Raw entities such as users, hosts, domains, and processes are strings. Graph models cannot directly train on those strings as graph indices.

The graph-table builder creates the bridge from cybersecurity telemetry to graph ML inputs.

## Local Run Commands

Run ingestion:

```bash
python pipelines/ingest_lanl_sample.py --config configs/lanl_ingest.yaml
```

Build clean event tables:

```bash
python pipelines/build_lanl_event_tables.py --config configs/lanl_event_tables.yaml
```

Build graph tables:

```bash
python pipelines/build_lanl_graph_tables.py --config configs/lanl_graph_tables.yaml
```

## Test Commands

Test ingestion:

```bash
python -m pytest tests/test_ingest_lanl_sample.py -ra
```

Test event-table builder:

```bash
python -m pytest tests/test_build_lanl_event_tables.py -ra
```

Test graph-table builder:

```bash
python -m pytest tests/test_build_lanl_graph_tables.py -ra
```

Run all tests:

```bash
python -m pytest -ra
```

## DVC Guidance

Large LANL raw files and generated parquet outputs should be tracked with DVC, not committed directly to Git.

DVC-tracked paths:

```text
data/external/lanl
data/processed/lanl
```

Git should track:

```text
configs/lanl_ingest.yaml
configs/lanl_event_tables.yaml
configs/lanl_graph_tables.yaml
pipelines/ingest_lanl_sample.py
pipelines/build_lanl_event_tables.py
pipelines/build_lanl_graph_tables.py
tests/test_ingest_lanl_sample.py
tests/test_build_lanl_event_tables.py
tests/test_build_lanl_graph_tables.py
docs/lanl_graph_data_pipeline.md
data/external/lanl.dvc
data/processed/lanl.dvc
```

Git should not track:

```text
raw LANL .txt.gz files
generated parquet files
large local data payloads
```

After generating graph tables, update DVC:

```bash
dvc add data/processed/lanl
dvc status
git status
```

## CI Rule

CI should not require real LANL data.

CI validates ingestion, event-table construction, and graph-table construction using temporary fixtures created inside tests.

This keeps CI fast, reproducible, and independent of large local datasets.

## Current Graph Pipeline Outputs

The final graph-ready outputs are:

```text
data/processed/lanl/graph_nodes.parquet
data/processed/lanl/graph_edges.parquet
data/processed/lanl/graph_table_manifest.json
```

These outputs become the direct input to graph inspection and graph model construction.