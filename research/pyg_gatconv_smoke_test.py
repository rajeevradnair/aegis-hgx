import torch
from torch_geometric.nn import GATConv


# Fix initialization so repeated runs produce the same
# randomly initialized layer parameters.
torch.manual_seed(7)


# Node-feature matrix.
#
# Rows:
# 0 = A
# 1 = B
# 2 = C
#
# Shape: [num_nodes, input_features] = [3, 2]
x = torch.tensor(
    [
        [1.0, 0.0],  # Node A
        [0.0, 1.0],  # Node B
        [1.0, 1.0],  # Node C
    ],
    dtype=torch.float32,
)


# Each column is one directed edge:
#
# sender → receiver
#
# Shape: [2, num_edges] = [2, 7]
edge_index = torch.tensor(
    [
        [0, 1, 2, 0, 1, 1, 2],  # Senders
        [0, 0, 0, 1, 1, 2, 2],  # Receivers
    ],
    dtype=torch.long,
)


gat = GATConv(
    in_channels=2,
    out_channels=2,
    heads=2,
    concat=True,
    negative_slope=0.2,
    dropout=0.0,
    add_self_loops=False,
    bias=False,
)

print(gat)

gat.eval()
with torch.no_grad():
    new_node_representation, attention_info = gat(
        x,
        edge_index,
        return_attention_weights=True,
    )

used_edge_index, attention_weights = attention_info

print("\nInput shape:", x.shape)
print("\nEach attention head generates the new feature representation for each node")
print("Additionally, each attention head generates the attention weight to be assigned to each edge")
print("\nOutput shape:", new_node_representation.shape, f" i.e. ({new_node_representation.shape[0]} nodes with {x.shape[1]} transformed features each) generated twice from each of the 2 attention heads")
print("\nUsed edge_index shape:", used_edge_index.shape)
print("\nAttention shape:", attention_weights.shape, f" i.e. ({attention_weights.shape[0]} edges with 1 attention weight each) generated twice, once from each attention head")


node_names = ["A", "B", "C"]

print("\nAttention coefficients:")

for edge_position in range(used_edge_index.shape[1]):
    sender = int(used_edge_index[0, edge_position])
    receiver = int(used_edge_index[1, edge_position])

    head_weights = attention_weights[edge_position]

    print(
        f"Edge {node_names[sender]} → {node_names[receiver]} : "
        f"head 1 = {head_weights[0].item():.4f}, "
        f"head 2 = {head_weights[1].item():.4f}"
    )

