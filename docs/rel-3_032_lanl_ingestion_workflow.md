LANL Ingestion Workflow

Purpose
This workflow ingests the LANL Unified Host and Network Dataset files from the local data/external/lanl directory and writes one processed parquet file per LANL event family.

Phase
This is data preparation for future training and evaluation.
It is not inference.
No serving API or production prediction path is changed by this workflow.

Input Directory
data/external/lanl

Expected Input Files
auth.txt.gz
dns.txt.gz
flows.txt.gz
proc.txt.gz
redteam.txt.gz

Config File
configs/lanl_ingest.yaml

Pipeline Script
pipelines/ingest_lanl_sample.py

Processed Output Directory
data/processed/lanl

Expected Outputs
auth.parquet
dns.parquet
flows.parquet
proc.parquet
redteam.parquet
ingest_manifest.json

Design Decision
Each LANL source file is written to a separate parquet file because the files represent different event families.

auth contains authentication activity.
dns contains name-resolution activity.
flows contains network-flow activity.
proc contains process activity.
redteam contains ground-truth red-team activity.

A single combined parquet file would blur event-family semantics too early.

The current ingestion step should preserve the event-family boundary.
Later event-table and graph-table builders can normalize, join, or transform these files intentionally.

Metadata Columns
The ingestion pipeline can add:

row_number
source_file
event_family

These columns help preserve lineage from processed parquet rows back to the original LANL source file.

Why This Matters
The graph phase needs entity and relationship data.

CIC remains the tabular baseline evidence source.
LANL becomes the primary graph and temporal modeling source.

This workflow is the first step toward:

LANL event tables
LANL graph node tables
LANL graph edge tables
NetworkX graph inspection
PyTorch Geometric graph objects
GCN, GraphSAGE, GAT, graph autoencoder, and temporal graph baselines

Local Run Command
python pipelines/ingest_lanl_sample.py --config configs/lanl_ingest.yaml

Focused Test Command
python -m pytest tests/test_ingest_lanl_sample.py -ra

Full Test Command
python -m pytest -ra

CI Rule
CI should not require the real LANL dataset.

The real LANL files are large and should not be committed to GitHub.
CI validates the ingestion logic using temporary gzipped LANL-style files created inside the test.

This keeps CI fast, reproducible, and independent of local private data.