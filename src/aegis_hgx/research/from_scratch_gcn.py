from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        input_features_count: int,
        output_features_count: int,
    ) -> None:
        super().__init__()

        # This is the learnable weight matrix for the layer.
        #
        # Shape:
        #   [input_features_count, output_features_count]
        #
        # Example:
        #   if input_features_count = 3
        #   and output_features_count = 2
        #   then weight shape = [3, 2]
        self.weight = nn.Parameter(
            torch.randn(
                input_features_count,
                output_features_count,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        
        # input -> GCN layer -> logits

        # Remember the forward pass formulka: 
        # Normalized adjacency matrix @ node feature matrix @ weight matrix
        # example shape: (4 4).       @ (4, 3).             @ (3, 2)
        #              resultant output (4, 2)

        # x is the node feature matrix.
        #
        # Shape:
        #   [num_nodes, input_features_count]

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
    input_features_count = x.shape[1]

    # Choose a small hidden dimension for this demo.
    #
    # This means each node will be converted from 3 features
    # into 2 learned hidden features (perhaps logits)
    output_features_count = 2

    layer = ManualGCNLayer(
        input_features_count=input_features_count,
        output_features_count=output_features_count,
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


class TinyManualGCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_classes: int,
    ) -> None:
        super().__init__()

        # First manual GCN layer.
        #
        # Purpose:
        #   Convert raw node features into hidden node representations.
        #
        # Shape:
        #   [num_nodes, input_features]
        #   -> [num_nodes, hidden_features]
        self.layer1 = ManualGCNLayer(
            input_features_count=input_features,
            output_features_count=hidden_features,
        )

        # Second manual GCN layer.
        #
        # Purpose:
        #   Convert hidden node representations into class scores.
        #
        # Shape:
        #   [num_nodes, hidden_features]
        #   -> [num_nodes, output_classes]
        self.layer2 = ManualGCNLayer(
            input_features_count=hidden_features,
            output_features_count=output_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        
        # GCN layer -> RELU -> GCN layer -> logits

        # First message-passing layer.
        #
        # Each node receives information from itself and its neighbors,
        # then the layer applies a learnable weight matrix.
        hidden = self.layer1(
            x=x,
            normalized_adjacency=normalized_adjacency,
        )

        # ReLU activation.
        #
        # This adds nonlinearity.
        # Without ReLU, stacking two linear GCN layers would still behave
        # like one larger linear transformation.
        hidden = F.relu(hidden)

        # Second message-passing layer.
        #
        # This produces raw class scores for each node.
        # These raw scores are called logits.
        logits = self.layer2(
            x=hidden,
            normalized_adjacency=normalized_adjacency,
        )

        # Shape:
        #   [num_nodes, output_classes]
        return logits


def run_tiny_manual_gcn_demo(
    x: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # Number of input features per node.
    #
    # If x shape is [4, 3],
    # then input_features = 3.
    input_features = x.shape[1]

    # Small hidden size for learning/demo purposes.
    hidden_features = 20

    # Number of output classes.
    #
    # If labels are 0 and 1,
    # then output_classes = 2.
    output_classes = int(y.max().item()) + 1

    model = TinyManualGCN(
        input_features=input_features,
        hidden_features=hidden_features,
        output_classes=output_classes,
    )

    # Run a forward pass.
    #
    # This is not training yet.
    # We are only checking that the model can produce logits.
    logits = model(
        x=x,
        normalized_adjacency=normalized_adjacency,
    )

    print()
    print("Tiny manual GCN model summary")
    print(
        {
            "model parameters": [(param.shape, param) for param in model.parameters()],
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


def train_tiny_manual_gcn(
    x: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    y: torch.Tensor,
    ) -> TinyManualGCN:

    # Make results reproducible.
    # This helps us see similar training behavior each time we run the script.
    torch.manual_seed(42)

    # Read the number of input features from x.
    #
    # Example:
    #   x shape = [4, 3]
    #   input_features = 3
    input_features = x.shape[1]

    # Choose a small hidden size for this learning lab.
    hidden_features = 4

    # Read the number of output classes from y.
    #
    # Example:
    #   y values = [0, 1]
    #   output_classes = 2
    output_classes = int(y.max().item()) + 1

    # Build the tiny two-layer manual GCN.
    model = TinyManualGCN(
        input_features=input_features,
        hidden_features=hidden_features,
        output_classes=output_classes,
    )

    # Adam updates the learnable weights inside our two manual GCN layers.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    # Save copies of the initial weights.
    # We will compare these against the final weights after training.
    # If the weights changed, optimizer.step() actually updated the model.
    initial_layer1_weight = model.layer1.weight.detach().clone()
    initial_layer2_weight = model.layer2.weight.detach().clone()

    # Number of full training passes.
    epochs = 100

    print()
    print("Training tiny manual GCN")

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------
        # TRAINING MODE
        # ------------------------------------------------------------
        # This is good habit, even though our tiny model has no dropout.
        model.train()

        # ------------------------------------------------------------
        # CLEAR OLD GRADIENTS
        # ------------------------------------------------------------
        # PyTorch accumulates gradients by default.
        # So before each new training step, we reset them.
        optimizer.zero_grad()

        # ------------------------------------------------------------
        # FORWARD PASS
        # ------------------------------------------------------------
        # The model uses:
        #   x                    -> node features
        #   normalized_adjacency -> graph structure after normalization
        #
        # Output:
        #   logits -> raw class scores for every node
        #
        # Shape:
        #   logits = [num_nodes, output_classes]
        logits = model(
            x=x,
            normalized_adjacency=normalized_adjacency,
        )

        # ------------------------------------------------------------
        # LOSS CALCULATION
        # ------------------------------------------------------------
        # Cross entropy compares:
        #   logits -> model predictions
        #   y      -> true node labels
        #
        # Important:
        #   We pass raw logits, not softmax probabilities.
        #   F.cross_entropy internally applies log-softmax.
        loss = F.cross_entropy(
            logits,
            y,
        )

        # ------------------------------------------------------------
        # BACKWARD PASS
        # ------------------------------------------------------------
        # This computes gradients for:
        #   model.layer1.weight
        #   model.layer2.weight
        loss.backward()

        # ------------------------------------------------------------
        # GRADIENT INSPECTION
        # ------------------------------------------------------------
        # After loss.backward(), PyTorch fills .grad for each learnable parameter.
        #
        # If these gradients are missing or zero, the layer is not learning.
        layer1_gradient_norm = model.layer1.weight.grad.norm().item()
        layer2_gradient_norm = model.layer2.weight.grad.norm().item()

        # ------------------------------------------------------------
        # PARAMETER UPDATE
        # ------------------------------------------------------------
        # The optimizer uses the gradients to update the weights.
        optimizer.step()

        # ------------------------------------------------------------
        # SIMPLE TRAINING ACCURACY
        # ------------------------------------------------------------
        # Pick the class with the largest logit for each node.
        predictions = logits.argmax(dim=1)

        # Compare predictions to true labels.
        accuracy = (
            predictions == y
        ).float().mean().item()

        # Print occasionally so we can see learning progress.
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "accuracy": float(accuracy),
                    "layer1_gradient_norm": layer1_gradient_norm,
                    "layer2_gradient_norm": layer2_gradient_norm,
                }
            )

    print()
    print("Final predictions")
    print(predictions)

    print()
    print("True labels")
    print(y)

    # ------------------------------------------------------------
    # WEIGHT UPDATE INSPECTION
    # ------------------------------------------------------------
    # Gradients prove that learning signal reached the weights.
    # Weight deltas prove that optimizer.step() actually changed the weights.
    layer1_weight_change = (
        model.layer1.weight.detach() - initial_layer1_weight
    ).norm().item()

    layer2_weight_change = (
        model.layer2.weight.detach() - initial_layer2_weight
    ).norm().item()

    print()
    print("Weight update summary")
    print(
        {
            "layer1_weight_change_norm": float(layer1_weight_change),
            "layer2_weight_change_norm": float(layer2_weight_change),
        }
    )

    return model



def main() -> None:

    # Important remember from scratch GCN is not ingesting edge_index. 
    # It takes node feature matrix, normalized adjacency matrix, and output labels 
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

    run_tiny_manual_gcn_demo(
        x=x,
        normalized_adjacency=normalized_adjacency,
        y=y,
    )

    train_tiny_manual_gcn(
        x=x,
        normalized_adjacency=normalized_adjacency,
        y=y,
    )


if __name__ == "__main__":
    main()