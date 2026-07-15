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