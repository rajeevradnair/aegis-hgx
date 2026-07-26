from __future__ import annotations

import pandas as pd
import torch

from pipelines.build_lanl_hetero_graph import (
    build_hetero_graph,
    remove_label_leakage_edges,
)
from aegis_hgx.models.baselines.train_lanl_hetero_gnn import (
    HeterogeneousGraphSAGE,
)


def make_tiny_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    nodes = pd.DataFrame(
        [
            {
                "node_id": 0,
                "entity_type": "user",
                "entity_name": "U620@DOM1",
                "label": 1,
            },
            {
                "node_id": 1,
                "entity_type": "user",
                "entity_name": "U100@DOM1",
                "label": 0,
            },
            {
                "node_id": 2,
                "entity_type": "user",
                "entity_name": "U200@DOM1",
                "label": 0,
            },
            {
                "node_id": 3,
                "entity_type": "user",
                "entity_name": "U300@DOM1",
                "label": 1,
            },
            {
                "node_id": 4,
                "entity_type": "host",
                "entity_name": "C1003",
                "label": 1,
            },
            {
                "node_id": 5,
                "entity_type": "host",
                "entity_name": "C2000",
                "label": 0,
            },
            {
                "node_id": 6,
                "entity_type": "host",
                "entity_name": "C3000",
                "label": 0,
            },
            {
                "node_id": 7,
                "entity_type": "host",
                "entity_name": "C4000",
                "label": 1,
            },
            {
                "node_id": 8,
                "entity_type": "process",
                "entity_name": "process_12480",
                "label": 0,
            },
            {
                "node_id": 9,
                "entity_type": "host_or_domain",
                "entity_name": "DOM1",
                "label": 0,
            },
        ]
    )

    base = {
        "label": 0,
        "event_family": "authentication",
        "edge_type": "authentication:auth_logon",
        "event_result": "success",
        "source_file": "auth.txt.gz",
    }

    edges = pd.DataFrame(
        [
            {
                **base,
                "source_node_id": 0,
                "destination_node_id": 4,
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "event_type": "auth_logon",
                "timestamp": 100,
            },
            {
                **base,
                "source_node_id": 1,
                "destination_node_id": 5,
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "event_type": "auth_logon",
                "timestamp": 110,
            },
            {
                **base,
                "source_node_id": 0,
                "destination_node_id": 8,
                "source_entity_type": "user",
                "destination_entity_type": "process",
                "event_family": "process",
                "event_type": "process_start",
                "edge_type": "process:process_start",
                "event_result": "start",
                "source_file": "proc.txt.gz",
                "timestamp": 120,
            },
            {
                **base,
                "source_node_id": 4,
                "destination_node_id": 9,
                "source_entity_type": "host",
                "destination_entity_type": "host_or_domain",
                "event_family": "dns",
                "event_type": "dns_resolution",
                "edge_type": "dns:dns_resolution",
                "event_result": "observed",
                "source_file": "dns.txt.gz",
                "timestamp": 130,
            },
            {
                **base,
                "source_node_id": 4,
                "destination_node_id": 5,
                "source_entity_type": "host",
                "destination_entity_type": "host",
                "event_family": "network_flow",
                "event_type": "network_flow",
                "edge_type": "network_flow:network_flow",
                "event_result": "unknown",
                "source_file": "flows.txt.gz",
                "timestamp": 140,
            },
            {
                "source_node_id": 0,
                "destination_node_id": 4,
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "event_family": "redteam_ground_truth",
                "event_type": "redteam_activity",
                "edge_type": (
                    "redteam_ground_truth:redteam_activity"
                ),
                "timestamp": 150,
                "event_result": "confirmed_redteam",
                "label": 1,
                "source_file": "redteam.txt.gz",
            },
        ]
    )

    return nodes, edges


def test_redteam_edges_are_removed() -> None:
    _, edges = make_tiny_tables()

    safe_edges = remove_label_leakage_edges(edges)

    assert len(safe_edges) == 5
    assert safe_edges["label"].sum() == 0
    assert "redteam_activity" not in set(
        safe_edges["event_type"]
    )


def test_heterogeneous_graph_schema_and_reverse_edges() -> None:
    nodes, edges = make_tiny_tables()

    data = build_hetero_graph(
        nodes=nodes,
        edges=edges,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    assert set(data.node_types) == {
        "user",
        "host",
        "process",
        "host_or_domain",
    }

    assert (
        "user",
        "auth_logon",
        "host",
    ) in data.edge_types

    assert (
        "host",
        "rev_auth_logon",
        "user",
    ) in data.edge_types

    assert (
        "process",
        "rev_process_start",
        "user",
    ) in data.edge_types

    assert not any(
        "redteam" in relation.lower()
        for _, relation, _ in data.edge_types
    )

    for node_type in ("user", "host"):
        store = data[node_type]

        assert store.x.size(0) == store.num_nodes
        assert store.x.size(1) == 7

        assert not bool(
            (
                store.train_mask
                & store.val_mask
            ).any()
        )

        assert not bool(
            (
                store.train_mask
                & store.test_mask
            ).any()
        )

        assert torch.all(
            store.train_mask
            | store.val_mask
            | store.test_mask
        )


def test_heterogeneous_graphsage_output_shapes() -> None:
    nodes, edges = make_tiny_tables()

    data = build_hetero_graph(
        nodes=nodes,
        edges=edges,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    model = HeterogeneousGraphSAGE(
        metadata=data.metadata(),
        hidden_channels=8,
        dropout=0.0,
    )

    logits = model(
        data.x_dict,
        data.edge_index_dict,
    )

    assert logits["user"].shape == (
        data["user"].num_nodes,
    )

    assert logits["host"].shape == (
        data["host"].num_nodes,
    )

    assert torch.isfinite(
        logits["user"]
    ).all()

    assert torch.isfinite(
        logits["host"]
    ).all()