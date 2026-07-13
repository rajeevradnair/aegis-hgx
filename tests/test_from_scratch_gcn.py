from __future__ import annotations

import torch
import torch.nn.functional as F

from aegis_hgx.research.from_scratch_gcn import (
    ManualGCNLayer,
    TinyManualGCN,
    build_toy_graph,
    normalize_adjacency,
)


def test_normalized_adjacency_has_expected_shape_and_finite_values() -> None:
    # Build the tiny toy graph used by the from-scratch GCN lab.
    x, adjacency, y = build_toy_graph()

    # Normalize the adjacency matrix.
    normalized_adjacency = normalize_adjacency(adjacency)

    # The graph has one adjacency row and column per node.
    assert normalized_adjacency.shape == (4, 4)

    # The node feature matrix should have 4 nodes and 3 features.
    assert x.shape == (4, 3)

    # The labels should have one label per node.
    assert y.shape == (4,)

    # Normalization should not create NaN or infinite values.
    assert torch.isfinite(normalized_adjacency).all()

    # Adding self-loops should make the diagonal positive.
    assert torch.all(torch.diag(normalized_adjacency) > 0)


def test_manual_gcn_layer_produces_expected_hidden_shape() -> None:
    # Build graph tensors.
    x, adjacency, _ = build_toy_graph()
    normalized_adjacency = normalize_adjacency(adjacency)

    # Create one manual GCN layer.
    # It maps 3 input features to 2 output features.
    layer = ManualGCNLayer(
        input_features_count=3,
        output_features_count=2,
    )

    # Run one forward pass.
    hidden = layer(
        x=x,
        normalized_adjacency=normalized_adjacency,
    )

    # The graph still has 4 nodes.
    # Each node now has 2 hidden features.
    assert hidden.shape == (4, 2)

    # Output should be numerically valid.
    assert torch.isfinite(hidden).all()


def test_tiny_manual_gcn_forward_backward_and_weight_update() -> None:
    # Make the test deterministic.
    torch.manual_seed(42)

    # Build graph tensors.
    x, adjacency, y = build_toy_graph()
    normalized_adjacency = normalize_adjacency(adjacency)

    # Build a tiny two-layer manual GCN.
    model = TinyManualGCN(
        input_features=3,
        hidden_features=4,
        output_classes=2,
    )

    # Save initial weights so we can prove they changed.
    initial_layer1_weight = model.layer1.weight.detach().clone()
    initial_layer2_weight = model.layer2.weight.detach().clone()

    # Use Adam for one tiny optimization step.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    # Clear old gradients.
    optimizer.zero_grad()

    # Forward pass.
    logits = model(
        x=x,
        normalized_adjacency=normalized_adjacency,
    )

    # The model should produce one row per node and one column per class.
    assert logits.shape == (4, 2)

    # Compute supervised classification loss.
    loss = F.cross_entropy(
        logits,
        y,
    )

    # Backward pass computes gradients.
    loss.backward()

    # Both layers should receive gradients.
    assert model.layer1.weight.grad is not None
    assert model.layer2.weight.grad is not None

    assert model.layer1.weight.grad.norm().item() > 0
    assert model.layer2.weight.grad.norm().item() > 0

    # Optimizer step updates weights.
    optimizer.step()

    # Prove both layers changed after the optimizer update.
    layer1_weight_change = (
        model.layer1.weight.detach() - initial_layer1_weight
    ).norm().item()

    layer2_weight_change = (
        model.layer2.weight.detach() - initial_layer2_weight
    ).norm().item()

    assert layer1_weight_change > 0
    assert layer2_weight_change > 0