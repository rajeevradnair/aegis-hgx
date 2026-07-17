import torch
import torch.nn.functional as F


# Each row represents one node.
# Each column represents one input feature.
#
# Shape: [num_nodes, input_features] = [3, 2]
x = torch.tensor(
    [
        [1.0, 0.0],  # Node A
        [0.0, 1.0],  # Node B
        [1.0, 1.0],  # Node C
    ]
)

# One attention head's feature-transformation matrix.
#
# It converts each node from 2 input features
# into 2 transformed features.
#
# Shape: [input_features, output_features] = [2, 2]
w = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)

# One attention head's learned attention vector.
#
# The receiver vector has 2 values.
# The sender vector has 2 values.
# Concatenating them produces 4 values.
#
# Shape: [2 * output_features] = [4]
attention_vector = torch.tensor([1.0, 0.0, 0.0, 1.0])

print("x shape:", x.shape)
print("w shape:", w.shape)
print("attention vector shape:", attention_vector.shape)

# Apply the same transformation matrix to every node.
#
# [3 nodes, 2 input features]
# @
# [2 input features, 2 output features]
# =
# [3 nodes, 2 output features]
z = x @ w

print("transformed nodes:")
print(z)
print("z shape:", z.shape)

# Node A is the receiver.
#
# z[0] selects row 0.
#
# Shape before selection: [3, 2]
# Shape after selection:  [2]
receiver = z[0]

print("receiver A:", receiver)
print("receiver shape:", receiver.shape)

# These are the transformed representations of all nodes
# sending messages to A.
#
# A, B, and C all send to A.
#
# Shape: [num_incoming_edges, output_features] = [3, 2]
senders = z

print("senders:")
print(senders)
print("senders shape:", senders.shape)

# There are three incoming edges to A.
#
# Repeat receiver A once for each sender.
#
# Original receiver shape: [2]
# Repeated shape:          [3, 2]
receiver_repeated = receiver.unsqueeze(0).repeat(senders.shape[0], 1)

print("repeated receiver:")
print(receiver_repeated)
print("shape:", receiver_repeated.shape)

# Concatenate along the feature dimension.
#
# receiver_repeated shape: [3, 2]
# senders shape:           [3, 2]
#
# Result shape:            [3, 4]
pair_features = torch.cat(
    [receiver_repeated, senders],
    dim=1,
)

print("receiver-sender pairs:")
print(pair_features)
print("pair shape:", pair_features.shape)

# Each edge pair has 4 values.
# The attention vector also has 4 values.
#
# [3 edges, 4 pair features]
# @
# [4 attention parameters]
# =
# [3 edge scores]
raw_scores = pair_features @ attention_vector

print("raw attention scores:", raw_scores)
print("raw score shape:", raw_scores.shape)

# GAT applies LeakyReLU to each raw edge score.
#
# Shape remains [3].
activated_scores = F.leaky_relu(
    raw_scores,
    negative_slope=0.2,
)

print("scores after LeakyReLU:", activated_scores)

# All three senders compete to influence receiver A.
#
# Softmax is applied across the three incoming edges.
#
# Shape remains [3].
attention_weights = torch.softmax(
    activated_scores,
    dim=0,
)

print("attention weights:", attention_weights)
print("sum:", attention_weights.sum())

# attention_weights shape: [3]
# senders shape:           [3, 2]
#
# Convert attention weights to [3, 1]
# so each edge weight can multiply both features
# of its corresponding sender.
# ELEMENT-WISE MULTIPLICATION (*)
# Inherent broadcast happens on attention_weights.unsqueeze(1) whose shape is (num_nodes, 1)
# broadcasted to (num_nodes, output_features) for the elementwise multi to work
weighted_messages = attention_weights.unsqueeze(1) * senders #Element-wise multiplication


print("weighted messages:")
print(weighted_messages)
print("weighted message shape:", weighted_messages.shape)

# Sum across all incoming edges.
#
# Before sum: [3 incoming edges, 2 features]
# After sum:  [2 features]
new_a = weighted_messages.sum(dim=0)

print("new representation for node A:", new_a)
print("new A shape:", new_a.shape)


# Each column represents one directed edge.
#
# First row:  sender node indices
# Second row: receiver node indices
#
# Node indices:
# A = 0
# B = 1
# C = 2
#
# Shape: [2, num_edges] = [2, 7]
edge_index = torch.tensor(
    [
        [0, 1, 2, 0, 1, 1, 2],  # senders
        [0, 0, 0, 1, 1, 2, 2],  # receivers
    ]
)

print("edge_index:")
print(edge_index)
print("edge_index shape:", edge_index.shape)

# edge_index[0] contains all sender indices.
# edge_index[1] contains all receiver indices.
#
# Both shapes: [num_edges] = [7]
sender_index = edge_index[0]
receiver_index = edge_index[1]

print("sender indices:", sender_index)
print("receiver indices:", receiver_index)

# Select one sender vector for each graph edge.
#
# z shape:             [3 nodes, 2 features]
# sender_index shape:  [7 edges]
# sender_z shape:      [7 edges, 2 features]
sender_z = z[sender_index]

print("sender vectors:")
print(sender_z)
print("sender_z shape:", sender_z.shape)

# Select one receiver vector for each graph edge.
#
# receiver_z shape: [7 edges, 2 features]
receiver_z = z[receiver_index]

print("receiver vectors:")
print(receiver_z)
print("receiver_z shape:", receiver_z.shape)

# For every edge, concatenate:
#
# [receiver features || sender features]
#
# receiver_z shape: [7, 2]
# sender_z shape:   [7, 2]
#
# pair_features shape: [7, 4]
all_pair_features = torch.cat(
    [receiver_z, sender_z],
    dim=1,
)

print("all receiver-sender pairs:")
print(all_pair_features)
print("shape:", all_pair_features.shape)

# Each edge row contains 4 values.
# The attention vector contains 4 values.
#
# [7 edges, 4 values]
# @
# [4 values]
# =
# [7 edge scores]

all_raw_scores = all_pair_features @ attention_vector

all_activated_scores = F.leaky_relu(
    all_raw_scores,
    negative_slope=0.2,
)

print("raw edge scores:", all_raw_scores)
print("activated scores:", all_activated_scores)

# Create space for one normalized coefficient per edge.
#
# Shape: [7 edges]
attention_weights_all = torch.zeros_like(all_activated_scores)

num_nodes = z.shape[0]

for receiver_node in range(num_nodes):
    # Find which edges enter this receiver.
    #
    # Example for receiver A:
    # receiver_index == 0
    #
    # Result:
    # [True, True, True, False, False, False, False]
    incoming_mask = receiver_index == receiver_node

    # Select only scores for edges entering this node.
    incoming_scores = all_activated_scores[incoming_mask]

    # Normalize only those incoming edge scores.
    incoming_weights = torch.softmax(
        incoming_scores,
        dim=0,
    )

    # Place the normalized values back into their edge positions.
    attention_weights_all[incoming_mask] = incoming_weights

print("all attention weights:", attention_weights_all)

for receiver_node in range(num_nodes):
    incoming_mask = receiver_index == receiver_node

    receiver_weight_sum = attention_weights_all[incoming_mask].sum()

    print(
        f"receiver {receiver_node} attention sum:",
        receiver_weight_sum.item(),
    )

# attention_weights_all shape: [7]
#
# Add one feature dimension:
# [7] → [7, 1]
#
# sender_z shape: [7, 2]
#
# Broadcasting:
# [7, 1] * [7, 2] → [7, 2]
all_weighted_messages = (
    attention_weights_all.unsqueeze(1) * sender_z
)

print("all weighted messages:")
print(all_weighted_messages)
print("shape:", all_weighted_messages.shape)

# Start every node output at zero.
#
# Shape: [3 nodes, 2 output features]
new_node_features = torch.zeros_like(z)

for edge_position in range(edge_index.shape[1]):
    # Determine which node receives this edge's message.
    receiver_node = receiver_index[edge_position]

    # Add this weighted message to that receiver.
    new_node_features[receiver_node] += all_weighted_messages[edge_position]

print("new node representations:")
print(new_node_features)
print("shape:", new_node_features.shape)

print(new_node_features[0])

 # Two independent feature-transformation matrices.
#
# Shape:
# [num_heads, input_features, output_features]
# =
# [2, 2, 2]
w_heads = torch.tensor(
    [
        # Head 1
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],

        # Head 2
        [
            [1.0, 1.0],
            [1.0, 0.0],
        ],
    ]
)

# One independent attention vector per head.
#
# Each vector scores:
# [receiver features || sender features]
#
# Shape:
# [num_heads, 2 * output_features]
# =
# [2, 4]
attention_vectors = torch.tensor(
    [
        [1.0, 0.0, 0.0, 1.0],  # Head 1
        [0.0, 1.0, 1.0, 0.0],  # Head 2
    ]
)

print("w_heads shape:", w_heads.shape)
print("attention_vectors shape:", attention_vectors.shape)


num_heads = w_heads.shape[0]
num_nodes = x.shape[0]

head_outputs = []
head_attention_weights = []

for head in range(num_heads):
    # Select this head's independent parameters.
    #
    # w_head shape: [2 input features, 2 output features]
    # a_head shape: [4 concatenated pair features]
    w_head = w_heads[head]
    a_head = attention_vectors[head]

    # Transform the complete node matrix using this head.
    #
    # [3, 2] @ [2, 2] = [3, 2]
    z_head = x @ w_head

    print(f"\nHead {head + 1}")
    print("transformed node matrix:")
    print(z_head)

    # Gather one transformed sender vector per edge.
    #
    # Shape: [7 edges, 2 features]
    sender_z = z_head[sender_index]

    # Gather one transformed receiver vector per edge.
    #
    # Shape: [7 edges, 2 features]
    receiver_z = z_head[receiver_index]

    # Build one receiver-sender row per edge.
    #
    # [7, 2] concatenated with [7, 2]
    # becomes [7, 4]
    pair_features = torch.cat(
        [receiver_z, sender_z],
        dim=1,
    )

    print("pair feature shape:", pair_features.shape)

    # Score every graph edge using this head's attention vector.
    #
    # [7 edges, 4 pair features]
    # @
    # [4 attention parameters]
    # =
    # [7 edge scores]
    raw_scores = pair_features @ a_head

    activated_scores = F.leaky_relu(
        raw_scores,
        negative_slope=0.2,
    )

    print("raw scores:", raw_scores)

    # One normalized coefficient per edge for this head.
    #
    # Shape: [7 edges]
    attention_weights = torch.zeros_like(activated_scores)

    for receiver_node in range(num_nodes):
        # Find all edges entering this receiver.
        incoming_mask = receiver_index == receiver_node

        # Get this head's scores for only that receiver.
        incoming_scores = activated_scores[incoming_mask]

        # Normalize those competing incoming edges.
        incoming_weights = torch.softmax(
            incoming_scores,
            dim=0,
        )

        # Return them to their original edge positions.
        attention_weights[incoming_mask] = incoming_weights

    print("attention weights:", attention_weights)

        # Weight each transformed sender vector.
    #
    # [7, 1] * [7, 2] = [7, 2]
    weighted_messages = (
        attention_weights.unsqueeze(1) * sender_z
    )

    # Allocate one output row per node.
    #
    # Shape: [3 nodes, 2 output features]
    head_output = torch.zeros_like(z_head)

    # Sum every edge message into its receiver.
    for edge_position in range(edge_index.shape[1]):
        receiver_node = receiver_index[edge_position]

        head_output[receiver_node] += weighted_messages[edge_position]

    print("head output:")
    print(head_output)

    # Save this head's results.
    head_outputs.append(head_output)
    head_attention_weights.append(attention_weights)

print("\nHead 1 final output:")
print(head_outputs[0])

print("\nHead 2 final output:")
print(head_outputs[1])

# Join the feature columns from all heads.
#
# Head 1: [3 nodes, 2 features]
# Head 2: [3 nodes, 2 features]
#
# Concatenated:
# [3 nodes, 4 total features]
multi_head_output = torch.cat(
    head_outputs,
    dim=1,
)

print("\nConcatenated multi-head output:")
print(multi_head_output)
print("multi-head output shape:", multi_head_output.shape)

# Each list item has shape [num_edges].
#
# Stack along dimension 1:
#
# [7] and [7]
# become
# [7 edges, 2 heads]
attention_by_edge_and_head = torch.stack(
    head_attention_weights,
    dim=1,
)

print("\nAttention per edge and head:")
print(attention_by_edge_and_head)
print(
    "attention tensor shape:",
    attention_by_edge_and_head.shape,
)

