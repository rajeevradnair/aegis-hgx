LANL Heterogeneous GraphSAGE Cheat Sheet
Core Task

Predict red-team participation for:

user nodes
host nodes

Use as context:

process nodes
host_or_domain nodes
Continuing Example
U620@DOM1 ──auth_logon──────► C1003
U620@DOM1 ──process_start────► process_12480
C1003      ──dns_resolution──► DOM1
C1003      ──network_flow────► C2000

The explicit red-team edge is removed from model input.

Heterogeneous Graph
Node type:
    what kind of entity is this?

Relation type:
    what kind of interaction connects the entities?

Canonical relation:

(
    source_node_type,
    relation_name,
    destination_node_type,
)

Example:

("user", "auth_logon", "host")
Main PyG Data
data["user"].x
data["host"].x

data[
    "user",
    "auth_logon",
    "host",
].edge_index

Useful dictionaries:

data.x_dict
data.edge_index_dict
Local Index Rule

Every node type has its own index space.

user local index 0

and:

host local index 0

refer to different entities.

Reverse Relations
data = T.ToUndirected(
    merge=False,
)(data)

Example:

user ──auth_logon─────► host
host ──rev_auth_logon─► user

Reverse relations enable message flow in both directions while preserving relation meaning.

Model
Typed node features
        ↓
Relation-specific SAGEConv
        ↓
HeteroConv
        ↓
Typed node embeddings
        ↓
User and host classification heads
SAGEConv

One operator handles one relation:

SAGEConv(
    (-1, -1),
    hidden_channels,
)

(-1, -1) means PyG infers source and destination feature dimensions.

HeteroConv
HeteroConv(
    relation_modules,
    aggr="sum",
)

It runs the matching relation operator and combines messages arriving at the same destination type.

Shapes
user.x          [num_users, 7]
host.x          [num_hosts, 7]

edge_index      [2, num_relation_edges]

user embedding  [num_users, hidden_channels]
host embedding  [num_hosts, hidden_channels]

user logits     [num_users]
host logits     [num_hosts]
Training Loss
binary_cross_entropy_with_logits(
    logits[train_mask],
    labels[train_mask],
    pos_weight=positive_weight,
)

Use raw logits during training.

Do not apply sigmoid before BCEWithLogitsLoss.

Training Flow
forward pass
    ↓
user loss + host loss
    ↓
backpropagation
    ↓
optimizer update
    ↓
validation PR-AUC
    ↓
save best checkpoint
Inference Flow
load best checkpoint
    ↓
model.eval()
    ↓
forward pass without gradients
    ↓
sigmoid(logits)
    ↓
apply validation-selected threshold
    ↓
alerts
Leakage Prevention

Exclude graph edges associated with:

redteam_ground_truth
redteam_activity
confirmed_redteam
redteam.txt.gz
edge label 1

Node labels remain the target.

Main Metrics
PR-AUC:
    ranking quality under class imbalance

ROC-AUC:
    overall positive-versus-negative ranking

Recall:
    fraction of positives detected

Precision:
    fraction of alerts that are positive

Alert count:
    operational workload
Honest Interpretation

The model answers:

Does this user or host appear risky based on its typed static neighborhood?

It does not yet answer:

Will this entity become malicious in a future time window?

Main Limitation
static transductive split

The complete graph structure is visible while node labels are split.

Chronological, leakage-safe temporal evaluation is deferred.