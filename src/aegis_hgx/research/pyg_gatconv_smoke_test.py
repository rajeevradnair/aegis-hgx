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
        f"Edge {node_names[receiver]} <- {node_names[sender]} : "
        f"head 1 = {head_weights[0].item():.4f}, "
        f"head 2 = {head_weights[1].item():.4f}"
    )


print("\nAttention sums by receiver:")

num_nodes = x.shape[0]

for receiver in range(num_nodes):
    # Select all edges entering this receiver.
    incoming_mask = used_edge_index[1] == receiver
    print(incoming_mask)

    # Shape before summing:
    # [number_of_incoming_edges, num_heads]
    incoming_attention = attention_weights[incoming_mask]
    print(incoming_attention)

    # Sum over incoming edges.
    #
    # Shape:
    # [number_of_incoming_edges, num_heads]
    # →
    # [num_heads]
    sums_per_head = incoming_attention.sum(dim=0)

    print(
        f"Receiver {node_names[receiver]}: "
        f"{sums_per_head.tolist()}"
    )


print("\nLearned parameter shapes:")

for parameter_name, parameter in gat.named_parameters():
    print(
        parameter_name,
        tuple(parameter.shape),
    )

print("\nIf W1 is the stored weight matrix for attention head 1 with shape [2, 2],")
print("and W2 is the stored weight matrix for attention head 2 with shape [2, 2],")
print("then lin.weight is W1 and W2 stacked vertically:")
print("lin.weight = torch.cat([W1, W2], dim=0)")
print("Therefore lin.weight has shape [4, 2].")
print("Internally, PyTorch calculates: X @ lin.weight.T")

import torch
import torch.nn.functional as F


# ---------------------------------------------------------
# Choose one edge and one attention head to inspect.
#
# Edge: B → A
# B = source/sender      j = 1
# A = destination        i = 0
#
# Inspect head 1:
# Python index 0
# ---------------------------------------------------------
source_node = 1       # j = B
destination_node = 0  # i = A
head = 0              # Head 1


# ---------------------------------------------------------
# PyG performs one large linear projection for all heads.
#
# gat.lin(x):
# [num_nodes, in_channels]
#     →
# [num_nodes, heads * out_channels]
#
# Then reshape:
# [num_nodes, heads, out_channels]
# ---------------------------------------------------------
projected = gat.lin(x)

z = projected.view(
    x.shape[0],
    gat.heads,
    gat.out_channels,
)

print("=" * 70)
print("ORIGINAL GAT ATTENTION EQUATION")
print("=" * 70)

print(
    "e_ij = LeakyReLU("
    "a^T [z_i || z_j]"
    ")"
)

print()
print("Definitions:")
print("j = source/sender node")
print("i = destination/receiver node")
print("z_j = transformed source/sender representation")
print("z_i = transformed destination/receiver representation")

print()
print(f"Selected edge: node {source_node} → node {destination_node}")
print(f"Selected head: {head + 1}")

print()
print("Projected node tensor shape:")
print("z.shape =", z.shape)
print(
    "[num_nodes, num_heads, features_per_head]"
)


# ---------------------------------------------------------
# Select transformed receiver and sender vectors.
#
# Each has shape:
# [out_channels]
# ---------------------------------------------------------
z_i = z[destination_node, head]
z_j = z[source_node, head]

print()
print("-" * 70)
print("TRANSFORMED ENDPOINT REPRESENTATIONS")
print("-" * 70)

print(f"z_i: transformed receiver node {destination_node}")
print(z_i)
print("z_i.shape =", z_i.shape)

print()

print(f"z_j: transformed sender node {source_node}")
print(z_j)
print("z_j.shape =", z_j.shape)

print()
print("If each transformed node has two features:")
print("z_i = [z_i1, z_i2]")
print("z_j = [z_j1, z_j2]")


# ---------------------------------------------------------
# Explicitly concatenate receiver then sender.
#
# [z_i || z_j]
#
# If each has 2 features:
# [2] concatenated with [2] → [4]
# ---------------------------------------------------------
concatenated_z = torch.cat(
    [z_i, z_j],
    dim=0,
)

print()
print("-" * 70)
print("EXPLICIT RECEIVER-SENDER CONCATENATION")
print("-" * 70)

print("[z_i || z_j] =")
print(concatenated_z)

print(
    "[z_i || z_j] = "
    "[z_i1, z_i2, z_j1, z_j2]"
)

print(
    "concatenated_z.shape =",
    concatenated_z.shape,
)


# ---------------------------------------------------------
# PyG stores the attention vector in two pieces:
#
# att_dst: receiver/destination scoring vector
# att_src: sender/source scoring vector
#
# Each shape for one head:
# [out_channels]
# ---------------------------------------------------------
a_dst = gat.att_dst[0, head]
a_src = gat.att_src[0, head]

print()
print("-" * 70)
print("PYG ATTENTION PARAMETERS")
print("-" * 70)

print("a_dst: scores the receiver/destination portion")
print(a_dst)
print("a_dst.shape =", a_dst.shape)

print()

print("a_src: scores the sender/source portion")
print(a_src)
print("a_src.shape =", a_src.shape)

print()
print("Conceptually:")
print("a_dst = [a1, a2]")
print("a_src = [a3, a4]")


# ---------------------------------------------------------
# Reconstruct the original full attention vector.
#
# a = [a_dst || a_src]
#
# If each part has 2 values:
# [a1, a2, a3, a4]
# ---------------------------------------------------------
full_attention_vector = torch.cat(
    [a_dst, a_src],
    dim=0,
)

print()
print("-" * 70)
print("RECONSTRUCTED ORIGINAL ATTENTION VECTOR")
print("-" * 70)

print("a = [a_dst || a_src]")
print("a =")
print(full_attention_vector)

print(
    "a = [a1, a2, a3, a4]"
)

print(
    "a.shape =",
    full_attention_vector.shape,
)


# ---------------------------------------------------------
# Method 1:
# Original concatenated GAT calculation.
#
# a^T [z_i || z_j]
# ---------------------------------------------------------
score_using_concatenation = torch.dot(
    full_attention_vector,
    concatenated_z,
)

print()
print("-" * 70)
print("METHOD 1: ORIGINAL CONCATENATED CALCULATION")
print("-" * 70)

print(
    "score = a^T [z_i || z_j]"
)

print(
    "score_using_concatenation =",
    score_using_concatenation.item(),
)


# ---------------------------------------------------------
# Method 2:
# PyG split calculation.
#
# a_dst^T z_i + a_src^T z_j
# ---------------------------------------------------------
destination_contribution = torch.dot(
    a_dst,
    z_i,
)

source_contribution = torch.dot(
    a_src,
    z_j,
)

score_using_split_form = (
    destination_contribution
    + source_contribution
)

print()
print("-" * 70)
print("METHOD 2: PYG SPLIT CALCULATION")
print("-" * 70)

print(
    "destination contribution = "
    "a_dst^T z_i"
)
print(
    "a_dst^T z_i =",
    destination_contribution.item(),
)

print()

print(
    "source contribution = "
    "a_src^T z_j"
)
print(
    "a_src^T z_j =",
    source_contribution.item(),
)

print()

print(
    "score = a_dst^T z_i "
    "+ a_src^T z_j"
)

print(
    "score_using_split_form =",
    score_using_split_form.item(),
)


# ---------------------------------------------------------
# Verify mathematical equivalence.
# ---------------------------------------------------------
print()
print("-" * 70)
print("EQUIVALENCE CHECK")
print("-" * 70)

print(
    "a^T [z_i || z_j] =",
    score_using_concatenation.item(),
)

print(
    "a_dst^T z_i + a_src^T z_j =",
    score_using_split_form.item(),
)

print(
    "Are they equal?",
    torch.allclose(
        score_using_concatenation,
        score_using_split_form,
    ),
)


# ---------------------------------------------------------
# Apply LeakyReLU.
#
# This gives the unnormalized edge score e_ij.
# ---------------------------------------------------------
e_ij = F.leaky_relu(
    score_using_split_form,
    negative_slope=gat.negative_slope,
)

print()
print("-" * 70)
print("APPLY LEAKYRELU")
print("-" * 70)

print(
    "e_ij = LeakyReLU("
    "a_dst^T z_i + a_src^T z_j"
    ")"
)

print("e_ij =", e_ij.item())


# ---------------------------------------------------------
# Obtain final normalized attention coefficients.
#
# PyG computes these for every edge and every head.
#
# Shape:
# [num_edges, num_heads]
# ---------------------------------------------------------
output, attention_info = gat(
    x,
    edge_index,
    return_attention_weights=True,
)

used_edge_index, edge_attention = attention_info

print()
print("-" * 70)
print("FINAL NORMALIZED EDGE ATTENTION")
print("-" * 70)

print(
    "edge_attention.shape =",
    edge_attention.shape,
)

print(
    "[num_edges, num_heads]"
)


# Find the row corresponding to source_node → destination_node.
matching_edges = (
    (used_edge_index[0] == source_node)
    &
    (used_edge_index[1] == destination_node)
)

matching_positions = matching_edges.nonzero(
    as_tuple=False
).flatten()

if matching_positions.numel() == 0:
    print(
        f"Edge {source_node} → "
        f"{destination_node} was not found."
    )
else:
    edge_position = matching_positions[0].item()

    alpha_ij = edge_attention[
        edge_position,
        head,
    ]

    print()
    print(
        f"Edge position: {edge_position}"
    )

    print(
        f"Final alpha_ij for edge "
        f"{source_node} → {destination_node}, "
        f"head {head + 1}:"
    )

    print(alpha_ij.item())


print()
print("=" * 70)
print("CORE INTERPRETATION")
print("=" * 70)

print(
    "att_dst scores the receiver portion z_i."
)

print(
    "att_src scores the sender portion z_j."
)

print(
    "att_dst and att_src are learned parameters, "
    "not final edge-attention coefficients."
)

print(
    "The final edge-attention tensor has shape "
    "[E, H]."
)