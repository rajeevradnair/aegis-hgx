# LANL Event Tables

## Purpose

This document summarizes the LANL event-table workflow in AEGIS-HGX.

The goal of this workflow is to convert processed LANL parquet files into clean, normalized event tables that can later be used for graph construction, graph inspection, and graph neural network training.

This is offline data preparation for future training and evaluation. It is not inference.

## Inputs

The workflow reads processed LANL parquet files from:

```text
data/processed/lanl
```

Expected input files:

```text
auth.parquet
dns.parquet
flows.parquet
proc.parquet
redteam.parquet
```

## Output Tables

The workflow writes:

```text
clean_auth_events.parquet
clean_dns_events.parquet
clean_flow_events.parquet
clean_process_events.parquet
clean_redteam_events.parquet
clean_all_events.parquet
event_table_manifest.json
```

## Canonical Event Schema

The combined event table uses the following shared columns:

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

## Event Family Mapping

Authentication events are normalized as user-to-host events.

DNS events are normalized as host-to-host-or-domain events.

Network-flow events are normalized as host-to-host events.

Process events are normalized as user-to-process events while preserving host context.

Red-team records are preserved as confirmed red-team events with label `1`.

Non-red-team events use the default label `0`.

## Why This Matters

Graph models need consistent source and destination entity columns.

The clean event table is the bridge between raw LANL telemetry and graph-ready node and edge tables.

The pipeline sequence is:

```text
processed LANL parquet files
        ↓
clean LANL event tables
        ↓
graph node and edge tables
        ↓
NetworkX inspection
        ↓
PyTorch Geometric graph objects
        ↓
GCN / GraphSAGE / GAT / temporal graph models
```

## Script

```text
pipelines/build_lanl_event_tables.py
```

## Config

```text
configs/lanl_event_tables.yaml
```

## Local Run Command

```bash
python pipelines/build_lanl_event_tables.py --config configs/lanl_event_tables.yaml
```

## Test Command

```bash
python -m pytest tests/test_build_lanl_event_tables.py -ra
```

## DVC Note

Generated parquet files should be tracked with DVC through:

```text
data/processed/lanl.dvc
```

Git should track code, configs, tests, documentation, CI files, and DVC pointer metadata.

Git should not track generated parquet files directly.
