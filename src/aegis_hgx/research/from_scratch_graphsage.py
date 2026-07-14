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
    
    # We will store one neighbor-mean vector per node.
    #
    # Shape should match x:
    #   [num_nodes, num_node_features]
    neighbor_means = torch.zeros_like(x)

    # Loop over every node in the graph.
    for node_id, neighbor_ids in neighbors.items():
        # If a node has no neighbors, keep its neighbor summary as zeros.
        #
        # In real systems, we may choose a different strategy.
        if not neighbor_ids:
            continue

        # Select feature rows for this node's neighbors.
        #
        # If neighbor_ids = [0, 2, 3],
        # then neighbor_features shape is:
        #   [3, num_node_features]
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
        # x contains the node's own features.
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
    # Read the number of input features from x.
    #
    # If x shape is [4, 3],
    # then input_features = 3.
    input_features = x.shape[1]

    # Choose a small output size for the demo.
    #
    # This means each node goes from 3 input features
    # to 2 learned hidden features.
    output_features = 2

    layer = ManualGraphSAGELayer(
        input_features=input_features,
        output_features=output_features,
    )

    # Run one manual GraphSAGE layer.
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
        # Purpose:
        #   Convert raw node features into hidden node embeddings.
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
        # Purpose:
        #   Convert hidden node embeddings into class logits.
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
        # x contains current node features.
        #
        # Shape:
        #   [num_nodes, input_features]

        # ------------------------------------------------------------
        # FIRST GRAPH SAGE LAYER
        # ------------------------------------------------------------

        # Compute mean neighbor features from the current x.
        #
        # Shape:
        #   [num_nodes, input_features]
        neighbor_means_layer1 = mean_aggregate_neighbors(
            x=x,
            neighbors=neighbors,
        )

        # Apply the first manual GraphSAGE layer.
        #
        # Shape:
        #   [num_nodes, input_features]
        #   -> [num_nodes, hidden_features]
        hidden = self.layer1(
            x=x,
            neighbor_means=neighbor_means_layer1,
        )

        # ReLU adds nonlinearity.
        #
        # Without ReLU, stacked linear layers are still mostly linear.
        hidden = F.relu(hidden)

        # ------------------------------------------------------------
        # SECOND GRAPH SAGE LAYER
        # ------------------------------------------------------------

        # Important:
        #   After layer 1, each node now has a hidden embedding.
        #
        # So for layer 2, neighbor aggregation must happen over hidden,
        # not over the original x.
        #
        # Shape:
        #   [num_nodes, hidden_features]
        neighbor_means_layer2 = mean_aggregate_neighbors(
            x=hidden,
            neighbors=neighbors,
        )

        # Apply the second manual GraphSAGE layer.
        #
        # Shape:
        #   [num_nodes, hidden_features]
        #   -> [num_nodes, output_classes]
        logits = self.layer2(
            x=hidden,
            neighbor_means=neighbor_means_layer2,
        )

        # logits are raw class scores.
        #
        # Shape:
        #   [num_nodes, output_classes]
        return logits



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



if __name__ == "__main__":
    main()