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
    output, attention_info = gat(
        x,
        edge_index,
        return_attention_weights=True,
    )

used_edge_index, attention_weights = attention_info

print("\nInput shape:", x.shape)
print("Output shape:", output.shape)
print("Used edge_index shape:", used_edge_index.shape)
print("Attention shape:", attention_weights.shape)