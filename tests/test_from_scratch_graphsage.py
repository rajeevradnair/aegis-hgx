from __future__ import annotations

import torch
import torch.nn.functional as F

from aegis_hgx.research.from_scratch_graphsage import (
    ManualGraphSAGELayer,
    TinyManualGraphSAGE,
    build_toy_graph,
    mean_aggregate_neighbors,
)


def test_mean_aggregate_neighbors_has_expected_values() -> None:
    x, neighbors, y = build_toy_graph()

    neighbor_means = mean_aggregate_neighbors(
        x=x,
        neighbors=neighbors,
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float,
    )

    assert x.shape == (4, 3)
    assert y.shape == (4,)
    assert neighbor_means.shape == (4, 3)
    assert torch.allclose(
        neighbor_means,
        expected,
        atol=1e-6,
    )


def test_manual_graphsage_layer_produces_expected_shape() -> None:
    x, neighbors, _ = build_toy_graph()

    neighbor_means = mean_aggregate_neighbors(
        x=x,
        neighbors=neighbors,
    )

    layer = ManualGraphSAGELayer(
        input_features=3,
        output_features=2,
    )

    hidden = layer(
        x=x,
        neighbor_means=neighbor_means,
    )

    assert hidden.shape == (4, 2)
    assert torch.isfinite(hidden).all()


def test_tiny_manual_graphsage_forward_backward_and_weight_update() -> None:
    torch.manual_seed(42)

    x, neighbors, y = build_toy_graph()

    model = TinyManualGraphSAGE(
        input_features=3,
        hidden_features=4,
        output_classes=2,
    )

    initial_layer1_self_weight = model.layer1.self_weight.detach().clone()
    initial_layer1_neighbor_weight = model.layer1.neighbor_weight.detach().clone()
    initial_layer2_self_weight = model.layer2.self_weight.detach().clone()
    initial_layer2_neighbor_weight = model.layer2.neighbor_weight.detach().clone()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    optimizer.zero_grad()

    logits = model(
        x=x,
        neighbors=neighbors,
    )

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()

    loss = F.cross_entropy(
        logits,
        y,
    )

    loss.backward()

    assert model.layer1.self_weight.grad is not None
    assert model.layer1.neighbor_weight.grad is not None
    assert model.layer2.self_weight.grad is not None
    assert model.layer2.neighbor_weight.grad is not None

    assert torch.isfinite(model.layer1.self_weight.grad).all()
    assert torch.isfinite(model.layer1.neighbor_weight.grad).all()
    assert torch.isfinite(model.layer2.self_weight.grad).all()
    assert torch.isfinite(model.layer2.neighbor_weight.grad).all()

    assert model.layer1.self_weight.grad.norm().item() > 0
    assert model.layer1.neighbor_weight.grad.norm().item() > 0
    assert model.layer2.self_weight.grad.norm().item() > 0
    assert model.layer2.neighbor_weight.grad.norm().item() > 0

    optimizer.step()

    layer1_self_weight_change = (
        model.layer1.self_weight.detach() - initial_layer1_self_weight
    ).norm().item()

    layer1_neighbor_weight_change = (
        model.layer1.neighbor_weight.detach() - initial_layer1_neighbor_weight
    ).norm().item()

    layer2_self_weight_change = (
        model.layer2.self_weight.detach() - initial_layer2_self_weight
    ).norm().item()

    layer2_neighbor_weight_change = (
        model.layer2.neighbor_weight.detach() - initial_layer2_neighbor_weight
    ).norm().item()

    assert layer1_self_weight_change > 0
    assert layer1_neighbor_weight_change > 0
    assert layer2_self_weight_change > 0
    assert layer2_neighbor_weight_change > 0