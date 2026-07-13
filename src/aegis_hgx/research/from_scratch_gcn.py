from __future__ import annotations

import torch
import torch.nn as nn


def build_toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            [1.0, 0.0, 0.0],  # node 0: user-like node
            [0.0, 1.0, 0.0],  # node 1: host-like node
            [0.0, 0.0, 1.0],  # node 2: process-like node
            [1.0, 1.0, 0.0],  # node 3: domain-like node
        ],
        dtype=torch.float,
    )

    # Adjacency matrix.
    #
    # Shape:
    #   [num_nodes, num_nodes]
    #
    # A[i, j] = 1 means node i is connected to node j.
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )

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

    return x, adjacency, y


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    # Number of nodes in the graph.
    num_nodes = adjacency.shape[0]

    # Identity matrix represents self-loops.
    #
    # Adding this means every node keeps its own information
    # during message passing.
    identity = torch.eye(num_nodes)

    # Add self-loops to the adjacency matrix.
    adjacency_with_self_loops = adjacency + identity

    # Degree is the number of connections each node has,
    # including its self-loop.
    #
    # Shape:
    #   [num_nodes]
    degree = adjacency_with_self_loops.sum(dim=1)

    # Compute 1 / sqrt(degree).
    #
    # This is used to prevent high-degree nodes from dominating.
    degree_inv_sqrt = torch.pow(degree, -0.5)

    # Convert the degree vector into a diagonal matrix.
    #
    # Shape:
    #   [num_nodes, num_nodes]
    degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt)

    # Symmetric GCN normalization:
    #
    #   A_norm = D^(-1/2) A D^(-1/2)
    #
    # This creates a normalized adjacency matrix.
    normalized_adjacency = (
        degree_inv_sqrt_matrix
        @ adjacency_with_self_loops
        @ degree_inv_sqrt_matrix
    )

    return normalized_adjacency


class ManualGCNLayer(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
    ) -> None:
        super().__init__()

        # This is the learnable weight matrix for the layer.
        #
        # Shape:
        #   [input_features, output_features]
        #
        # Example:
        #   if input_features = 3
        #   and output_features = 2
        #   then weight shape = [3, 2]
        self.weight = nn.Parameter(
            torch.randn(
                input_features,
                output_features,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        # x is the node feature matrix.
        #
        # Shape:
        #   [num_nodes, input_features]

        # normalized_adjacency is the graph mixing matrix.
        #
        # Shape:
        #   [num_nodes, num_nodes]

        # First matrix multiplication:
        #
        #   normalized_adjacency @ x
        #
        # This mixes each node's features with its neighbors' features.
        #
        # Shape:
        #   [num_nodes, num_nodes] @ [num_nodes, input_features]
        #   = [num_nodes, input_features]
        mixed_neighbor_features = normalized_adjacency @ x

        # Second matrix multiplication:
        #
        #   mixed_neighbor_features @ self.weight
        #
        # This applies a learnable transformation.
        #
        # Shape:
        #   [num_nodes, input_features] @ [input_features, output_features]
        #   = [num_nodes, output_features]
        output = mixed_neighbor_features @ self.weight

        return output


def run_one_manual_gcn_layer_demo(
    x: torch.Tensor,
    normalized_adjacency: torch.Tensor,
) -> torch.Tensor:
    # Read the number of input features from x.
    #
    # If x shape is [4, 3],
    # then input_features = 3.
    input_features = x.shape[1]

    # Choose a small hidden dimension for this demo.
    #
    # This means each node will be converted from 3 features
    # into 2 learned hidden features (perhaps logits)
    output_features = 2

    layer = ManualGCNLayer(
        input_features=input_features,
        output_features=output_features,
    )

    # Run one FORWARD PASS of manual GCN layer.
    hidden = layer(
        x=x,
        normalized_adjacency=normalized_adjacency,
    )

    print()
    print("One manual GCN layer summary")
    print(
        {
            "input_x_shape": list(x.shape),
            "normalized_adjacency_shape": list(normalized_adjacency.shape),
            "weight_shape": list(layer.weight.shape),
            "hidden_output_shape": list(hidden.shape),
        }
    )

    print()
    print("Hidden node representations - logits")
    print(hidden)

    return hidden


def main() -> None:

    x, adjacency, y = build_toy_graph()

    normalized_adjacency = normalize_adjacency(adjacency)

    print()
    print("Sample graph tensors")
    print(
        {
            "x_shape": list(x.shape),
            "adjacency_shape": list(adjacency.shape),
            "normalized_adjacency_shape": list(normalized_adjacency.shape),
            "y_shape": list(y.shape),
        }
    )

    print()
    print("Normalized adjacency")
    print(normalized_adjacency)

    run_one_manual_gcn_layer_demo(
        x=x,
        normalized_adjacency=normalized_adjacency,
    )


if __name__ == "__main__":
    main()