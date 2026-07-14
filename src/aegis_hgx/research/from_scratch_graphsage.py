from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_toy_graph() -> tuple[torch.Tensor, dict[int, list[int]], torch.Tensor]:
    # Node feature matrix.
    #
    # Shape:
    #   [num_nodes, num_node_features]
    #
    # Here:
    #   4 nodes
    #   3 features per node
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # user-like node
            [0.0, 1.0, 0.0],  # host-like node
            [0.0, 0.0, 1.0],  # process-like node
            [1.0, 1.0, 0.0],  # domain-like node
        ],
        dtype=torch.float,
    )

    # Adjacency list.
    #
    # Each key is a node ID.
    # Each value is the list of neighbor node IDs.
    #
    # This toy graph is undirected for easier learning.
    neighbors = {
        0: [1],
        1: [0, 2, 3],
        2: [1],
        3: [1],
    }

    # Node labels.
    #
    # Shape:
    #   [num_nodes]
    #
    # 0 = benign
    # 1 = suspicious
    y = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )

    return x, neighbors, y


def mean_aggregate_neighbors(
    x: torch.Tensor,
    neighbors: dict[int, list[int]],
) -> torch.Tensor:
    # Store one mean-neighbor vector per node.
    #
    # Shape:
    #   [num_nodes, num_node_features]
    neighbor_means = torch.zeros_like(x)

    # Loop over every node in the adjacency list.
    for node_id, neighbor_ids in neighbors.items():
        # If a node has no neighbors, keep its neighbor summary as zeros.
        if not neighbor_ids:
            continue

        # Select feature rows for this node's neighbors.
        #
        # Example:
        #   neighbor_ids = [0, 2, 3]
        #   x[neighbor_ids] shape = [3, num_node_features]
        neighbor_features = x[neighbor_ids]

        # Average neighbor features.
        #
        # Shape:
        #   [num_node_features]
        neighbor_mean = neighbor_features.mean(dim=0)

        # Store this node's neighbor summary.
        neighbor_means[node_id] = neighbor_mean

    return neighbor_means


class ManualGraphSAGELayer(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
    ) -> None:
        super().__init__()

        # This weight transforms the node's own features.
        #
        # Shape:
        #   [input_features, output_features]
        self.self_weight = nn.Parameter(
            torch.randn(
                input_features,
                output_features,
            )
        )

        # This weight transforms the mean of the node's neighbor features.
        #
        # Shape:
        #   [input_features, output_features]
        self.neighbor_weight = nn.Parameter(
            torch.randn(
                input_features,
                output_features,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbor_means: torch.Tensor,
    ) -> torch.Tensor:
        # x contains each node's own features.
        #
        # Shape:
        #   [num_nodes, input_features]

        # neighbor_means contains one mean-neighbor vector per node.
        #
        # Shape:
        #   [num_nodes, input_features]

        # Transform each node's own features.
        #
        # Shape:
        #   [num_nodes, input_features] @ [input_features, output_features]
        #   = [num_nodes, output_features]
        self_part = x @ self.self_weight

        # Transform each node's neighbor summary.
        #
        # Shape:
        #   [num_nodes, input_features] @ [input_features, output_features]
        #   = [num_nodes, output_features]
        neighbor_part = neighbor_means @ self.neighbor_weight

        # Combine own-node signal and neighbor signal.
        #
        # Shape:
        #   [num_nodes, output_features]
        output = self_part + neighbor_part

        return output


def run_one_manual_graphsage_layer_demo(
    x: torch.Tensor,
    neighbor_means: torch.Tensor,
) -> torch.Tensor:
    # Number of input features per node.
    input_features = x.shape[1]

    # Small output size for the demo.
    output_features = 2

    layer = ManualGraphSAGELayer(
        input_features=input_features,
        output_features=output_features,
    )

    hidden = layer(
        x=x,
        neighbor_means=neighbor_means,
    )

    print()
    print("One manual GraphSAGE layer summary")
    print(
        {
            "x_shape": list(x.shape),
            "neighbor_means_shape": list(neighbor_means.shape),
            "self_weight_shape": list(layer.self_weight.shape),
            "neighbor_weight_shape": list(layer.neighbor_weight.shape),
            "hidden_shape": list(hidden.shape),
        }
    )

    print()
    print("Hidden node representations")
    print(hidden)

    return hidden


class TinyManualGraphSAGE(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_classes: int,
    ) -> None:
        super().__init__()

        # First manual GraphSAGE layer.
        #
        # Shape:
        #   [num_nodes, input_features]
        #   -> [num_nodes, hidden_features]
        self.layer1 = ManualGraphSAGELayer(
            input_features=input_features,
            output_features=hidden_features,
        )

        # Second manual GraphSAGE layer.
        #
        # Shape:
        #   [num_nodes, hidden_features]
        #   -> [num_nodes, output_classes]
        self.layer2 = ManualGraphSAGELayer(
            input_features=hidden_features,
            output_features=output_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: dict[int, list[int]],
    ) -> torch.Tensor:
        # ------------------------------------------------------------
        # FIRST GRAPH SAGE LAYER
        # ------------------------------------------------------------

        # Compute mean neighbor features from raw node features.
        #
        # Shape:
        #   [num_nodes, input_features]
        neighbor_means_layer1 = mean_aggregate_neighbors(
            x=x,
            neighbors=neighbors,
        )

        # First layer:
        #   raw node features + raw neighbor means -> hidden embeddings
        hidden = self.layer1(
            x=x,
            neighbor_means=neighbor_means_layer1,
        )

        # ReLU adds nonlinearity.
        hidden = F.relu(hidden)

        # ------------------------------------------------------------
        # SECOND GRAPH SAGE LAYER
        # ------------------------------------------------------------

        # Compute mean neighbor features from hidden embeddings.
        #
        # This is important:
        #   layer 2 aggregates learned embeddings,
        #   not raw features again.
        #
        # Shape:
        #   [num_nodes, hidden_features]
        neighbor_means_layer2 = mean_aggregate_neighbors(
            x=hidden,
            neighbors=neighbors,
        )

        # Second layer:
        #   hidden embeddings + hidden neighbor means -> class logits
        logits = self.layer2(
            x=hidden,
            neighbor_means=neighbor_means_layer2,
        )

        # Shape:
        #   [num_nodes, output_classes]
        return logits


def run_tiny_manual_graphsage_demo(
    x: torch.Tensor,
    neighbors: dict[int, list[int]],
    y: torch.Tensor,
) -> torch.Tensor:
    input_features = x.shape[1]
    hidden_features = 4
    output_classes = int(y.max().item()) + 1

    model = TinyManualGraphSAGE(
        input_features=input_features,
        hidden_features=hidden_features,
        output_classes=output_classes,
    )

    # Forward pass only.
    # No training happens here.
    logits = model(
        x=x,
        neighbors=neighbors,
    )

    print()
    print("Tiny manual GraphSAGE model summary")
    print(
        {
            "input_x_shape": list(x.shape),
            "hidden_features": hidden_features,
            "output_classes": output_classes,
            "logits_shape": list(logits.shape),
        }
    )

    print()
    print("Node logits")
    print(logits)

    return logits


def train_tiny_manual_graphsage(
    x: torch.Tensor,
    neighbors: dict[int, list[int]],
    y: torch.Tensor,
) -> TinyManualGraphSAGE:
    # Make the run reproducible.
    torch.manual_seed(42)

    input_features = x.shape[1]
    hidden_features = 4
    output_classes = int(y.max().item()) + 1

    model = TinyManualGraphSAGE(
        input_features=input_features,
        hidden_features=hidden_features,
        output_classes=output_classes,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    # Save initial weights so we can prove training changed them.
    initial_layer1_self_weight = model.layer1.self_weight.detach().clone()
    initial_layer1_neighbor_weight = model.layer1.neighbor_weight.detach().clone()
    initial_layer2_self_weight = model.layer2.self_weight.detach().clone()
    initial_layer2_neighbor_weight = model.layer2.neighbor_weight.detach().clone()

    epochs = 100

    print()
    print("Training tiny manual GraphSAGE")

    for epoch in range(1, epochs + 1):
        # Training mode.
        model.train()

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward pass.
        #
        # logits shape:
        #   [num_nodes, output_classes]
        logits = model(
            x=x,
            neighbors=neighbors,
        )

        # Cross-entropy compares raw logits against true labels.
        #
        # Important:
        #   do not apply softmax before F.cross_entropy.
        loss = F.cross_entropy(
            logits,
            y,
        )

        # Backward pass.
        #
        # This computes gradients for all four learned matrices:
        #   layer1.self_weight
        #   layer1.neighbor_weight
        #   layer2.self_weight
        #   layer2.neighbor_weight
        loss.backward()

        # Inspect gradient norms before optimizer.step().
        layer1_self_gradient_norm = model.layer1.self_weight.grad.norm().item()
        layer1_neighbor_gradient_norm = model.layer1.neighbor_weight.grad.norm().item()
        layer2_self_gradient_norm = model.layer2.self_weight.grad.norm().item()
        layer2_neighbor_gradient_norm = model.layer2.neighbor_weight.grad.norm().item()

        # Update weights.
        optimizer.step()

        # Simple training accuracy.
        predictions = logits.argmax(dim=1)

        accuracy = (
            predictions == y
        ).float().mean().item()

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "accuracy": float(accuracy),
                    "layer1_self_gradient_norm": float(layer1_self_gradient_norm),
                    "layer1_neighbor_gradient_norm": float(layer1_neighbor_gradient_norm),
                    "layer2_self_gradient_norm": float(layer2_self_gradient_norm),
                    "layer2_neighbor_gradient_norm": float(layer2_neighbor_gradient_norm),
                }
            )

    print()
    print("Final predictions")
    print(predictions)

    print()
    print("True labels")
    print(y)

    # Prove the optimizer actually changed each learned matrix.
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

    print()
    print("Weight update summary")
    print(
        {
            "layer1_self_weight_change_norm": float(layer1_self_weight_change),
            "layer1_neighbor_weight_change_norm": float(layer1_neighbor_weight_change),
            "layer2_self_weight_change_norm": float(layer2_self_weight_change),
            "layer2_neighbor_weight_change_norm": float(layer2_neighbor_weight_change),
        }
    )

    # Inference-style check.
    #
    # This does not update weights.
    model.eval()

    with torch.no_grad():
        final_logits = model(
            x=x,
            neighbors=neighbors,
        )

        final_probabilities = torch.softmax(
            final_logits,
            dim=1,
        )

    print()
    print("Final class probabilities")
    print(final_probabilities)

    return model


def main() -> None:
    x, neighbors, y = build_toy_graph()

    neighbor_means = mean_aggregate_neighbors(
        x=x,
        neighbors=neighbors,
    )

    print("Toy GraphSAGE tensors")
    print(
        {
            "x_shape": list(x.shape),
            "neighbor_means_shape": list(neighbor_means.shape),
            "y_shape": list(y.shape),
        }
    )

    print()
    print("Node features")
    print(x)

    print()
    print("Mean neighbor features")
    print(neighbor_means)

    run_one_manual_graphsage_layer_demo(
        x=x,
        neighbor_means=neighbor_means,
    )

    run_tiny_manual_graphsage_demo(
        x=x,
        neighbors=neighbors,
        y=y,
    )

    train_tiny_manual_graphsage(
        x=x,
        neighbors=neighbors,
        y=y,
    )


if __name__ == "__main__":
    main()