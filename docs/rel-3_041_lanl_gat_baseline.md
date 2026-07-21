# LANL GAT Baseline

## Purpose

This document records the implementation and evaluation contract for the LANL Graph Attention Network baseline in Aegis-HGX.

The baseline extends the existing homogeneous LANL graph experiments from GCN and GraphSAGE to an attention-based message-passing model. Its purpose is to test whether learned neighbor weighting improves suspicious-node classification on the same graph, split policy, metrics, and experiment-tracking foundation used by the other static graph baselines.

## Phase Classification

| Topic | Project Phase |
|---|---|
| GAT layer and head design | Architecture/design |
| Parameter optimization | Training |
| Best-checkpoint selection | Validation |
| Final held-out metrics | Test |
| Attention-head count and dropout choices | Research/experimentation |
| Loading the saved checkpoint for prediction | Inference |

## Implementation Files

```text
src/aegis_hgx/models/baselines/train_lanl_gat_pyg.py
configs/lanl_gat_pyg.yaml
```

Generated artifacts:

```text
artifacts/models/lanl/lanl_gat_model.pt
reports/lanl_gat_metrics.json
mlflow/mlflow.db
mlflow/mlruns/
```

## Input Graph Contract

The training pipeline loads a serialized homogeneous PyTorch Geometric `Data` object.

Required fields:

```text
data.x
data.edge_index
data.y
data.num_nodes
```

Expected tensor shapes:

```text
data.x          [num_nodes, num_node_features]
data.edge_index [2, num_edges]
data.y          [num_nodes]
```

Expected dtypes:

```text
data.x          floating point
data.edge_index torch.long
data.y          torch.long
```

The graph is validated before training for:

- required attributes,
- tensor ranks,
- node-count alignment,
- valid edge indices,
- finite node features,
- non-empty edge set,
- and valid class labels.

## Model Architecture

The model is a two-layer node-classification GAT.

```text
LANL node-feature matrix
        |
        v
Feature dropout
        |
        v
Multi-head GAT hidden layer
        |
        v
ELU activation
        |
        v
Feature dropout
        |
        v
Single-head GAT output layer
        |
        v
One raw score per class for every node
```

Default configuration:

```yaml
model:
  hidden_channels: 16
  heads: 4
  dropout: 0.30
  attention_dropout: 0.20
  negative_slope: 0.20
  add_self_loops: true
```

### Hidden-layer shape

For:

```text
input feature width = F
hidden_channels     = 16 per head
heads               = 4
```

the first layer performs:

```text
[num_nodes, F]
->
[num_nodes, 16 * 4]
->
[num_nodes, 64]
```

Every attention head receives the complete node-feature matrix and complete graph. The nodes are not divided among heads.

Each head owns an independent feature projection and independent attention parameters. The four head outputs are concatenated.

### Output-layer shape

For binary node classification:

```text
output_channels = 2
```

The second GAT layer performs:

```text
[num_nodes, 64]
->
[num_nodes, 2]
```

The output columns are raw class logits:

```text
column 0: normal-node score
column 1: suspicious-node score
```

No softmax is applied inside the model because `cross_entropy` consumes raw logits.

## GAT Attention Mechanics

For a directed edge from sender node `j` to receiver node `i`, each attention head:

1. Projects the complete node-feature matrix.
2. Retrieves the transformed sender and receiver vectors for each edge.
3. Calculates an unnormalized edge score.
4. Applies LeakyReLU.
5. Applies softmax over edges entering the same receiver.
6. Multiplies sender messages by the normalized coefficients.
7. Sums weighted messages into the receiver node.

Conceptually:

```text
edge_index determines which nodes may communicate
attention determines how strongly each eligible sender influences its receiver
```

Attention coefficients are edge-level values. With `E` graph edges and `H` heads, the conceptual attention tensor has shape:

```text
[E, H]
```

## Data Splitting

The implementation creates reproducible stratified node masks:

```text
train_mask [num_nodes]
val_mask   [num_nodes]
test_mask  [num_nodes]
```

Default ratios:

```text
train:      70%
validation: 15%
test:       15%
```

The split is stratified independently within each class so that rare suspicious nodes are distributed more consistently across the three sets.

The masks are validated for:

- correct shape,
- Boolean dtype,
- non-empty splits,
- no overlap,
- complete node coverage,
- and class presence warnings.

## Important Transductive Detail

Every forward pass uses:

```text
data.x
data.edge_index
```

for the complete graph.

However, supervised labels are restricted by mask:

```text
training loss   uses train_mask
model selection uses val_mask
final metrics   use test_mask
```

This is a static transductive node-classification baseline.

The current random stratified split is intentionally consistent with the preceding GCN and GraphSAGE baselines. It is not yet a chronological or temporal leakage-safe evaluation. Temporal splitting is handled in the later temporal-modeling phase.

## Class Imbalance Handling

Class weights are calculated from training labels only.

This avoids using validation or test label distributions to influence optimization.

Inverse-frequency weighting gives the rare class a larger loss contribution:

```text
rare suspicious class -> larger class weight
common normal class    -> smaller class weight
```

Training loss:

```python
F.cross_entropy(
    train_logits,
    train_labels,
    weight=class_weights,
)
```

## Training Flow

For every epoch:

1. Set `model.train()`.
2. Clear accumulated gradients.
3. Run a full-graph forward pass.
4. Select training-node logits.
5. Calculate weighted cross-entropy loss.
6. Backpropagate gradients.
7. Update parameters with Adam.
8. Set `model.eval()`.
9. Run validation with dropout disabled.
10. Record train and validation loss and accuracy.
11. Save the model state when validation loss improves.

Default optimizer configuration:

```yaml
training:
  epochs: 50
  learning_rate: 0.005
  weight_decay: 0.0005
```

## Best-Checkpoint Policy

The final epoch is not automatically treated as the best model.

The implementation stores the model state associated with the lowest validation loss.

The saved checkpoint and final test evaluation both use that best validation checkpoint.

This prevents a later overfit epoch from replacing a better earlier model.

## Evaluation Metrics

The best checkpoint is evaluated once on test nodes.

Recorded metrics:

- accuracy,
- precision,
- recall,
- F1,
- ROC-AUC,
- PR-AUC,
- positive test-node count,
- negative test-node count,
- total test-node count.

For rare cyber anomalies, PR-AUC, recall, precision, and F1 are generally more informative than accuracy alone.

A model that predicts every node as normal may have high accuracy but near-zero suspicious-node recall.

## Saved Model Checkpoint

The saved checkpoint contains:

- model state dictionary,
- model class,
- input width,
- hidden width per head,
- number of heads,
- concatenated hidden width,
- output width,
- feature dropout,
- attention dropout,
- LeakyReLU negative slope,
- self-loop policy,
- class weights,
- best validation epoch,
- full configuration,
- UTC save timestamp.

The best model state is stored on CPU so the checkpoint remains portable between CPU and GPU environments.

## Metrics Artifact

The JSON metrics artifact records:

- graph metadata,
- model architecture,
- training configuration,
- split configuration,
- split counts,
- class weights,
- full epoch history,
- best validation epoch,
- and final test metrics.

This artifact supports comparison against the LANL GCN and GraphSAGE baselines.

## MLflow Tracking

The implementation creates one MLflow run for the experiment.

Logged parameters include:

- model family,
- graph type,
- node and edge counts,
- feature count,
- split ratios and seed,
- hidden width per head,
- number of heads,
- concatenated hidden width,
- feature dropout,
- attention dropout,
- learning rate,
- weight decay,
- epoch count,
- device,
- and positive label.

Logged metrics include:

- per-epoch train loss,
- per-epoch train accuracy,
- per-epoch validation loss,
- per-epoch validation accuracy,
- best validation loss,
- best validation accuracy,
- final train metrics,
- and final test metrics.

Logged evidence artifacts include:

```text
config YAML
metrics JSON
model checkpoint
```

## Run Command

From the repository root:

```bash
python -m src.aegis_hgx.models.baselines.train_lanl_gat_pyg \
  --config configs/lanl_gat_pyg.yaml
```

## Expected Execution Checks

The run should confirm:

```text
graph loaded successfully
node and edge tensor shapes are valid
masks cover all nodes without overlap
hidden GAT width equals hidden_channels * heads
forward logits have shape [num_nodes, num_classes]
all logits are finite
training and validation losses are finite
best checkpoint is saved
test metrics are written
MLflow artifacts are logged
```

## Common Failure Modes

### Incorrect hidden width

With:

```text
hidden_channels = 16
heads = 4
concat = true
```

the second layer must receive 64 features, not 16.

### Edge direction error

PyG uses:

```text
edge_index[0] = sender indices
edge_index[1] = receiver indices
```

Reversing the rows changes the message-passing direction.

### Duplicate or unexpected self-loops

The model currently uses:

```text
add_self_loops = true
```

The graph-building pipeline should document whether self-loops already exist. Unexpected duplicate self-loops can alter aggregation behavior.

### Accuracy hides class collapse

High accuracy with low recall, F1, or PR-AUC usually indicates that the model predicts almost every node as normal.

### Missing positive nodes in a split

Very small positive-class counts can produce a validation or test split with only one class. ROC-AUC and PR-AUC may then be undefined.

### NaN or infinite loss

Check:

- node-feature finiteness,
- feature scaling,
- learning rate,
- graph corruption,
- and gradient instability.

### GAT memory cost

GAT calculates attention values per edge and per head. Increasing graph edges, head count, or hidden width can materially increase memory and runtime.

## Baseline Comparison Contract

The GAT result should be compared with GCN and GraphSAGE using:

- the same LANL graph artifact,
- the same split ratios,
- the same split seed,
- the same class-weighting policy,
- the same metric definitions,
- and equivalent checkpoint-selection rules.

The key research question is:

> Does learned neighbor weighting improve suspicious-node detection enough to justify the additional runtime and memory cost compared with GCN and GraphSAGE?

## Current Limitations

- Homogeneous graph only.
- Static graph only.
- Node-level classification only.
- Random stratified split rather than temporal split.
- No relation-specific message passing.
- No edge features in the attention calculation.
- No neighbor sampling.
- No attention-based explanation report.
- No multi-seed statistical comparison yet.
- No hyperparameter sweep yet.

1{
  "best_validation_epoch": {
    "epoch": 50.0,
    "train_accuracy": 0.9174866676330566,
    "train_loss": 0.42757073044776917,
    "val_accuracy": 0.9778638482093811,
    "val_loss": 0.3532688617706299
  },
  "generated_at_utc": "2026-07-17T01:25:56.371394+00:00",
  "graph": {
    "label_values": [
      0,
      1
    ],
    "node_feature_count": 8,
    "num_edges": 4000749,
    "num_nodes": 24102
  },
  "graph_path": "artifacts/models/lanl/lanl_homogeneous_graph.pt",
  "metrics_path": "reports/lanl_gat_metrics.json",
  "model": {
    "add_self_loops": true,
    "attention_dropout": 0.2,
    "class": "LanlGAT",
    "concatenated_hidden_width": 64,
    "dropout": 0.3,
    "heads": 4,
    "hidden_channels_per_head": 16,
    "input_channels": 8,
    "negative_slope": 0.2,
    "output_channels": 2
  },
  "model_path": "artifacts/models/lanl/lanl_gat_model.pt",
  "phase": "gat_training_and_evaluation",
  "split": {
    "seed": 42,
    "test_ratio": 0.15,
    "train_ratio": 0.7,
    "val_ratio": 0.15
  },
  "split_counts": {
    "test_nodes": 3618,
    "train_nodes": 16870,
    "val_nodes": 3614
  },
  "test_metrics": {
    "test_accuracy": 0.9778883360972913,
    "test_f1": 0.5652173913043478,
    "test_negative_nodes": 3556,
    "test_positive_nodes": 62,
    "test_pr_auc": 0.4386557969998165,
    "test_precision": 0.4262295081967213,
    "test_recall": 0.8387096774193549,
    "test_roc_auc": 0.9526742624913822,
    "test_total_nodes": 3618
  },
  "training": {
    "class_weights": [
      0.5085307955741882,
      29.805654525756836
    ],
    "epochs": 50,
    "learning_rate": 0.005,
    "weight_decay": 0.0005
  },
  "training_history": [
    {
      "epoch": 1.0,
      "train_accuracy": 0.48826318979263306,
      "train_loss": 0.7275345325469971,
      "val_accuracy": 0.049529608339071274,
      "val_loss": 0.7021975517272949
    },
    {
      "epoch": 2.0,
      "train_accuracy": 0.4671606421470642,
      "train_loss": 0.7189901471138,
      "val_accuracy": 0.1853901445865631,
      "val_loss": 0.6865949630737305
    },
    {
      "epoch": 3.0,
      "train_accuracy": 0.45198577642440796,
      "train_loss": 0.7040872573852539,
      "val_accuracy": 0.28832319378852844,
      "val_loss": 0.6722238659858704
    },
    {
      "epoch": 4.0,
      "train_accuracy": 0.5139893293380737,
      "train_loss": 0.687441885471344,
      "val_accuracy": 0.34975096583366394,
      "val_loss": 0.6586605310440063
    },
    {
      "epoch": 5.0,
      "train_accuracy": 0.6450504064559937,
      "train_loss": 0.6745325922966003,
      "val_accuracy": 0.3746541142463684,
      "val_loss": 0.6460484266281128
    },
    {
      "epoch": 6.0,
      "train_accuracy": 0.6743331551551819,
      "train_loss": 0.6513657569885254,
      "val_accuracy": 0.39762037992477417,
      "val_loss": 0.6342708468437195
    },
    {
      "epoch": 7.0,
      "train_accuracy": 0.6997036337852478,
      "train_loss": 0.6434118151664734,
      "val_accuracy": 0.9820144176483154,
      "val_loss": 0.623503565788269
    },
    {
      "epoch": 8.0,
      "train_accuracy": 0.7181386947631836,
      "train_loss": 0.6326229572296143,
      "val_accuracy": 0.9817376732826233,
      "val_loss": 0.613105297088623
    },
    {
      "epoch": 9.0,
      "train_accuracy": 0.7606402039527893,
      "train_loss": 0.638914167881012,
      "val_accuracy": 0.9820144176483154,
      "val_loss": 0.60272616147995
    },
    {
      "epoch": 10.0,
      "train_accuracy": 0.7475993037223816,
      "train_loss": 0.6096675395965576,
      "val_accuracy": 0.9820144176483154,
      "val_loss": 0.5925596952438354
    },
    {
      "epoch": 11.0,
      "train_accuracy": 0.7401304244995117,
      "train_loss": 0.5948343873023987,
      "val_accuracy": 0.9817376732826233,
      "val_loss": 0.5826393961906433
    },
    {
      "epoch": 12.0,
      "train_accuracy": 0.7432128190994263,
      "train_loss": 0.5944105386734009,
      "val_accuracy": 0.9814609885215759,
      "val_loss": 0.5727301836013794
    },
    {
      "epoch": 13.0,
      "train_accuracy": 0.7405453324317932,
      "train_loss": 0.6066053509712219,
      "val_accuracy": 0.9809075593948364,
      "val_loss": 0.5628742575645447
    },
    {
      "epoch": 14.0,
      "train_accuracy": 0.7524600028991699,
      "train_loss": 0.5750257968902588,
      "val_accuracy": 0.9803541898727417,
      "val_loss": 0.5529211759567261
    },
    {
      "epoch": 15.0,
      "train_accuracy": 0.7775340676307678,
      "train_loss": 0.5756236910820007,
      "val_accuracy": 0.9795240759849548,
      "val_loss": 0.542953372001648
    },
    {
      "epoch": 16.0,
      "train_accuracy": 0.7832246422767639,
      "train_loss": 0.5429811477661133,
      "val_accuracy": 0.9789706468582153,
      "val_loss": 0.5327141880989075
    },
    {
      "epoch": 17.0,
      "train_accuracy": 0.7974510788917542,
      "train_loss": 0.5476844906806946,
      "val_accuracy": 0.978693962097168,
      "val_loss": 0.5223646759986877
    },
    {
      "epoch": 18.0,
      "train_accuracy": 0.8164196610450745,
      "train_loss": 0.5744306445121765,
      "val_accuracy": 0.9651355743408203,
      "val_loss": 0.5121756792068481
    },
    {
      "epoch": 19.0,
      "train_accuracy": 0.8292234539985657,
      "train_loss": 0.5493987202644348,
      "val_accuracy": 0.9618151783943176,
      "val_loss": 0.5021699666976929
    },
    {
      "epoch": 20.0,
      "train_accuracy": 0.8477771282196045,
      "train_loss": 0.4863327443599701,
      "val_accuracy": 0.9551743268966675,
      "val_loss": 0.4920673966407776
    },
    {
      "epoch": 21.0,
      "train_accuracy": 0.8666864037513733,
      "train_loss": 0.5567899346351624,
      "val_accuracy": 0.9460431933403015,
      "val_loss": 0.4822418689727783
    },
    {
      "epoch": 22.0,
      "train_accuracy": 0.8870776295661926,
      "train_loss": 0.5093830823898315,
      "val_accuracy": 0.940785825252533,
      "val_loss": 0.47256359457969666
    },
    {
      "epoch": 23.0,
      "train_accuracy": 0.8927682042121887,
      "train_loss": 0.5482120513916016,
      "val_accuracy": 0.9346984028816223,
      "val_loss": 0.46353259682655334
    },
    {
      "epoch": 24.0,
      "train_accuracy": 0.8935388326644897,
      "train_loss": 0.49418455362319946,
      "val_accuracy": 0.9261206388473511,
      "val_loss": 0.45479461550712585
    },
    {
      "epoch": 25.0,
      "train_accuracy": 0.891464114189148,
      "train_loss": 0.48683393001556396,
      "val_accuracy": 0.9192031025886536,
      "val_loss": 0.4465683102607727
    },
    {
      "epoch": 26.0,
      "train_accuracy": 0.880497932434082,
      "train_loss": 0.5230720043182373,
      "val_accuracy": 0.9156059622764587,
      "val_loss": 0.4390730857849121
    },
    {
      "epoch": 27.0,
      "train_accuracy": 0.8772969841957092,
      "train_loss": 0.4879077672958374,
      "val_accuracy": 0.9164360761642456,
      "val_loss": 0.4312049448490143
    },
    {
      "epoch": 28.0,
      "train_accuracy": 0.8959099054336548,
      "train_loss": 0.5048831105232239,
      "val_accuracy": 0.9175428748130798,
      "val_loss": 0.42390960454940796
    },
    {
      "epoch": 29.0,
      "train_accuracy": 0.8930646181106567,
      "train_loss": 0.5228093862533569,
      "val_accuracy": 0.9186496734619141,
      "val_loss": 0.41722893714904785
    },
    {
      "epoch": 30.0,
      "train_accuracy": 0.8966212272644043,
      "train_loss": 0.5151821374893188,
      "val_accuracy": 0.9205865859985352,
      "val_loss": 0.4111585319042206
    },
    {
      "epoch": 31.0,
      "train_accuracy": 0.906994640827179,
      "train_loss": 0.4244201183319092,
      "val_accuracy": 0.9283342361450195,
      "val_loss": 0.40485841035842896
    },
    {
      "epoch": 32.0,
      "train_accuracy": 0.9060462117195129,
      "train_loss": 0.46184688806533813,
      "val_accuracy": 0.9391255974769592,
      "val_loss": 0.39953842759132385
    },
    {
      "epoch": 33.0,
      "train_accuracy": 0.9237107038497925,
      "train_loss": 0.49445027112960815,
      "val_accuracy": 0.9521306157112122,
      "val_loss": 0.39518290758132935
    },
    {
      "epoch": 34.0,
      "train_accuracy": 0.9071725010871887,
      "train_loss": 0.4697343111038208,
      "val_accuracy": 0.9612617492675781,
      "val_loss": 0.39131617546081543
    },
    {
      "epoch": 35.0,
      "train_accuracy": 0.9189685583114624,
      "train_loss": 0.4394693672657013,
      "val_accuracy": 0.9764803647994995,
      "val_loss": 0.38787999749183655
    },
    {
      "epoch": 36.0,
      "train_accuracy": 0.9326615333557129,
      "train_loss": 0.43135538697242737,
      "val_accuracy": 0.9770337343215942,
      "val_loss": 0.38446760177612305
    },
    {
      "epoch": 37.0,
      "train_accuracy": 0.9275044202804565,
      "train_loss": 0.48856621980667114,
      "val_accuracy": 0.9770337343215942,
      "val_loss": 0.3805921673774719
    },
    {
      "epoch": 38.0,
      "train_accuracy": 0.9407824277877808,
      "train_loss": 0.42728307843208313,
      "val_accuracy": 0.9770337343215942,
      "val_loss": 0.3765392601490021
    },
    {
      "epoch": 39.0,
      "train_accuracy": 0.9307646751403809,
      "train_loss": 0.4507683515548706,
      "val_accuracy": 0.9770337343215942,
      "val_loss": 0.3730282783508301
    },
    {
      "epoch": 40.0,
      "train_accuracy": 0.9158861637115479,
      "train_loss": 0.44895339012145996,
      "val_accuracy": 0.9770337343215942,
      "val_loss": 0.3702020049095154
    },
    {
      "epoch": 41.0,
      "train_accuracy": 0.9301126003265381,
      "train_loss": 0.4165726602077484,
      "val_accuracy": 0.9770337343215942,
      "val_loss": 0.36754074692726135
    },
    {
      "epoch": 42.0,
      "train_accuracy": 0.9308832287788391,
      "train_loss": 0.41742604970932007,
      "val_accuracy": 0.9773104786872864,
      "val_loss": 0.3652403652667999
    },
    {
      "epoch": 43.0,
      "train_accuracy": 0.9362773895263672,
      "train_loss": 0.40220382809638977,
      "val_accuracy": 0.9778638482093811,
      "val_loss": 0.36392998695373535
    },
    {
      "epoch": 44.0,
      "train_accuracy": 0.9246591329574585,
      "train_loss": 0.4308667480945587,
      "val_accuracy": 0.9781405925750732,
      "val_loss": 0.3630315363407135
    },
    {
      "epoch": 45.0,
      "train_accuracy": 0.9338470697402954,
      "train_loss": 0.4773983061313629,
      "val_accuracy": 0.9781405925750732,
      "val_loss": 0.3605969250202179
    },
    {
      "epoch": 46.0,
      "train_accuracy": 0.9312388896942139,
      "train_loss": 0.4999317526817322,
      "val_accuracy": 0.9775871634483337,
      "val_loss": 0.35782068967819214
    },
    {
      "epoch": 47.0,
      "train_accuracy": 0.9303497076034546,
      "train_loss": 0.39893314242362976,
      "val_accuracy": 0.9778638482093811,
      "val_loss": 0.35615044832229614
    },
    {
      "epoch": 48.0,
      "train_accuracy": 0.9407824277877808,
      "train_loss": 0.5116060972213745,
      "val_accuracy": 0.9781405925750732,
      "val_loss": 0.3542995750904083
    },
    {
      "epoch": 49.0,
      "train_accuracy": 0.9273858666419983,
      "train_loss": 0.4894782602787018,
      "val_accuracy": 0.9778638482093811,
      "val_loss": 0.35347098112106323
    },
    {
      "epoch": 50.0,
      "train_accuracy": 0.9174866676330566,
      "train_loss": 0.42757073044776917,
      "val_accuracy": 0.9778638482093811,
      "val_loss": 0.3532688617706299
    }
  ]
}