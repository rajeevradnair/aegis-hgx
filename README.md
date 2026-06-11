# Aegis-HGX

Aegis-HGX, short for Aegis HyperGraph eXplainable Threat Detection, is a research-grade machine learning platform for cyber anomaly detection.

The project starts with simple tabular baselines and evolves into graph neural networks, heterogeneous graph models, temporal graph learning, explainability, adversarial robustness, and production-style ML infrastructure.

## Project Goal

The goal of Aegis-HGX is to detect suspicious cyber behavior by modeling security activity as relationships between entities such as users, hosts, processes, files, and network endpoints.

Instead of only asking whether one event row is suspicious, Aegis-HGX asks:

- Which entities are behaving abnormally?
- Which relationships are becoming risky?
- Which graph patterns suggest attack behavior?
- Can the system explain why an alert was produced?
- Can the model remain reliable under drift, imbalance, and adversarial behavior?

## Core Research Thesis

Cyber attacks often emerge through relationships over time.

A single login, process execution, file write, or network connection may appear harmless in isolation. However, the combination of relationships between users, hosts, processes, files, and external destinations can reveal suspicious behavior.

Aegis-HGX investigates whether graph-based and temporal graph-based models can improve anomaly detection compared with tabular baselines.

## System Direction

The project will be built incrementally:

1. Establish a clean research and engineering foundation.
2. Generate synthetic cyber event data.
3. Train simple tabular baselines.
4. Add experiment tracking, testing, and reproducibility.
5. Build graph representations from cyber events.
6. Train graph neural network baselines.
7. Add temporal modeling and relationship memory.
8. Evaluate robustness, drift, failure modes, and explainability.
9. Package the system as a serious ML research and engineering portfolio project.

## Initial Architecture

```text
Raw cyber events
    |
    v
Data pipelines
    |
    v
Feature engineering
    |
    +--------------------+
    |                    |
    v                    v
Tabular baselines     Graph models
    |                    |
    v                    v
Evaluation, metrics, threshold tuning
    |
    v
Inference service and monitoring
    |
    v
Reports, model cards, data cards, and research documentation
```

## Repository Structure

```text
aegis-hgx/
  docs/
    research_thesis.md
    architecture.md
    metrics_guide.md

  data/
    raw/
    interim/
    processed/
    external/

  pipelines/
    generate_synthetic_events.py
    ingest_public_dataset.py
    build_tabular_features.py
    build_graph_tables.py

  src/
    aegis_hgx/
      domain/
      features/
      models/
        baselines/
        graph/
        temporal/
        serving/
        evaluation/
        experiments/
        monitoring/
      utils/

  configs/
  tests/
  reports/
  infra/
  notebooks/
```

## Intended Portfolio Signal

Aegis-HGX is designed to demonstrate capability across:

- applied machine learning
- graph machine learning
- temporal modeling
- anomaly detection
- ML experimentation
- reproducibility
- model evaluation
- production-style ML infrastructure
- explainability
- performance and scalability analysis

