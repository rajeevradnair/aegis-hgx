from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_undirected,
    negative_sampling,
)
from datetime import datetime, timezone

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

DEFAULT_CONFIG_PATH = Path(
    "configs/lanl_graph_autoencoder.yaml"
)

def load_config(config_path: Path) -> dict[str, Any]:
    """Load the graph-autoencoder experiment configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a dictionary."
        )

    return config

def validate_config(config: dict[str, Any]) -> None:
    """Validate fields required by the baseline trainer."""

    required_sections = [
        "input",
        "output",
        "split",
        "model",
        "training",
        "evaluation",
        "experiment_tracking",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}"
            )

    val_ratio = float(config["split"]["val_ratio"])
    test_ratio = float(config["split"]["test_ratio"])

    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(
            "split.val_ratio must be between 0 and 1."
        )

    if test_ratio <= 0.0 or test_ratio >= 1.0:
        raise ValueError(
            "split.test_ratio must be between 0 and 1."
        )

    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            "Validation and test ratios must leave training edges."
        )

    hidden_channels = int(
        config["model"]["hidden_channels"]
    )
    latent_channels = int(
        config["model"]["latent_channels"]
    )

    if hidden_channels <= 0:
        raise ValueError(
            "model.hidden_channels must be positive."
        )

    if latent_channels <= 0:
        raise ValueError(
            "model.latent_channels must be positive."
        )

    dropout = float(config["model"]["dropout"])

    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(
            "model.dropout must be in the range [0, 1)."
        )

    negative_ratio = float(
        config["split"]["negative_sampling_ratio"]
    )

    if negative_ratio <= 0.0:
        raise ValueError(
            "split.negative_sampling_ratio must be positive."
        )
    

def set_random_seeds(seed: int) -> None:
    """Set random seeds used by splitting and model training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a graph autoencoder on the "
            "LANL homogeneous PyG graph."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML experiment configuration.",
    )

    return parser.parse_args()


def validate_graph(
    data: Data,
    expect_undirected: bool,
) -> None:
    """Validate tensors required for graph reconstruction."""

    if data.x is None:
        raise ValueError(
            "The graph does not contain data.x node features."
        )

    if data.edge_index is None:
        raise ValueError(
            "The graph does not contain data.edge_index."
        )

    # Node features must be a matrix:
    # [number_of_nodes, number_of_features]
    if data.x.ndim != 2:
        raise ValueError(
            "data.x must have shape [N, F_in], "
            f"but received {tuple(data.x.shape)}."
        )

    if data.x.size(0) == 0:
        raise ValueError(
            "data.x contains no nodes."
        )

    if data.x.size(1) == 0:
        raise ValueError(
            "data.x contains no node features."
        )

    if not torch.is_floating_point(data.x):
        raise TypeError(
            "data.x must use a floating-point dtype, "
            f"but received {data.x.dtype}."
        )

    if not torch.isfinite(data.x).all():
        raise ValueError(
            "data.x contains NaN or infinite values."
        )

    # PyG expects edge_index to have:
    # row 0 = source node IDs
    # row 1 = target node IDs
    if data.edge_index.ndim != 2:
        raise ValueError(
            "data.edge_index must be a two-dimensional tensor."
        )

    if data.edge_index.size(0) != 2:
        raise ValueError(
            "data.edge_index must have shape [2, E], "
            f"but received {tuple(data.edge_index.shape)}."
        )

    if data.edge_index.size(1) == 0:
        raise ValueError(
            "data.edge_index contains no graph edges."
        )

    if data.edge_index.dtype != torch.long:
        raise TypeError(
            "data.edge_index must use torch.long, "
            f"but received {data.edge_index.dtype}."
        )

    num_nodes = data.x.size(0)

    minimum_node_id = int(data.edge_index.min().item())
    maximum_node_id = int(data.edge_index.max().item())

    if minimum_node_id < 0:
        raise ValueError(
            "data.edge_index contains a negative node ID."
        )

    if maximum_node_id >= num_nodes:
        raise ValueError(
            "data.edge_index references a node outside data.x: "
            f"maximum node ID is {maximum_node_id}, "
            f"but data.x contains {num_nodes} nodes."
        )

    if expect_undirected and not data.is_undirected():
        print(
            "\nWarning: the stored LANL graph is directed or "
            "does not contain every reverse edge."
        )
        print(
            "A cloned graph will be symmetrized for the "
            "undirected graph-autoencoder baseline."
        )


def prepare_link_prediction_graph(
    data: Data,
    make_undirected: bool,
) -> Data:
    """Prepare a clean graph copy for edge reconstruction."""

    # Keep the original loaded graph unchanged.
    prepared_data = data.clone()

    # This graph may contain edge labels created by an earlier
    # pipeline or experiment.
    #
    # For this self-supervised GAE baseline, RandomLinkSplit
    # must create fresh binary labels:
    #
    # observed edge = 1
    # sampled non-edge = 0
    link_prediction_keys = [
        "edge_label",
        "edge_label_index",
        "pos_edge_label",
        "pos_edge_label_index",
        "neg_edge_label",
        "neg_edge_label_index",
    ]

    for key in link_prediction_keys:
        if key in prepared_data:
            value = prepared_data[key]

            if isinstance(value, torch.Tensor):
                value_description = (
                    f"shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}"
                )
            else:
                value_description = f"type={type(value).__name__}"

            print(
                f"Removing pre-existing {key}: "
                f"{value_description}"
            )

            del prepared_data[key]


    if prepared_data.edge_attr is not None:
        print("\nEdge attributes detected")
        print("------------------------")
        print(
            "edge_attr shape:",
            tuple(prepared_data.edge_attr.shape),
        )
        print(
            "edge_attr dtype:",
            prepared_data.edge_attr.dtype,
        )
        print(
            "This topology-only GAE will ignore edge attributes "
            "in its cloned working graph."
        )

        # Remove edge attributes only from the cloned graph.
        # The original saved LANL graph is not modified.
        prepared_data.edge_attr = None

    original_edge_count = prepared_data.edge_index.size(1)

    # Remove stored self-loops as reconstruction targets.
    edge_index, _ = remove_self_loops(
        prepared_data.edge_index
    )

    after_self_loop_count = edge_index.size(1)

    # Collapse repeated copies of the exact same directed edge.
    edge_index = coalesce(
        edge_index,
        num_nodes=prepared_data.num_nodes,
    )

    after_duplicate_removal_count = edge_index.size(1)

    # The default GAE inner-product decoder is symmetric.
    # Therefore, create an undirected structural view by adding
    # missing reverse edges to this cloned graph.
    if make_undirected:
        edge_index = to_undirected(
            edge_index,
            num_nodes=prepared_data.num_nodes,
        )

        # Ensure any duplicates introduced during conversion
        # are represented only once.
        edge_index = coalesce(
            edge_index,
            num_nodes=prepared_data.num_nodes,
        )

    prepared_data.edge_index = edge_index

    removed_self_loops = (
        original_edge_count - after_self_loop_count
    )

    removed_duplicates = (
        after_self_loop_count
        - after_duplicate_removal_count
    )

    reverse_edges_added = (
        edge_index.size(1)
        - after_duplicate_removal_count
    )

    print("\nLink-prediction graph preparation")
    print("---------------------------------")
    print(
        "Original directed edge entries:",
        f"{original_edge_count:,}",
    )
    print(
        "Removed self-loop entries:",
        f"{removed_self_loops:,}",
    )
    print(
        "Removed duplicate edge entries:",
        f"{removed_duplicates:,}",
    )
    print(
        "Reverse edge entries added:",
        f"{reverse_edges_added:,}",
    )
    print(
        "Prepared directed edge entries:",
        f"{edge_index.size(1):,}",
    )

    if make_undirected and not prepared_data.is_undirected():
        raise ValueError(
            "Failed to create a symmetric graph representation."
        )

    if "edge_label" in prepared_data:
        raise ValueError(
            "Pre-existing edge_label was not removed before "
            "self-supervised link splitting."
        )

    if "edge_label_index" in prepared_data:
        raise ValueError(
            "Pre-existing edge_label_index was not removed before "
            "self-supervised link splitting."
        )

    return prepared_data


def create_edge_splits(
    data: Data,
    config: dict[str, Any],
) -> tuple[Data, Data, Data]:
    """Split graph relationships into train, validation, and test."""

    splitter = RandomLinkSplit(
        num_val=float(
            config["split"]["val_ratio"]
        ),
        num_test=float(
            config["split"]["test_ratio"]
        ),

        # Prevent U1->H1 and H1->U1 from being
        # assigned to separate data splits.
        is_undirected=bool(
            config["split"]["is_undirected"]
        ),

        # Training negatives will be sampled dynamically
        # by GAE.recon_loss() during each epoch.
        add_negative_train_samples=False,

        # Validation and test receive fixed negative
        # samples for repeatable metric calculation.
        neg_sampling_ratio=float(
            config["split"]["negative_sampling_ratio"]
        ),

        # For this first baseline, training positive edges
        # also provide training message-passing context.
        disjoint_train_ratio=0.0,
    )

    train_data, val_data, test_data = splitter(data)

    return train_data, val_data, test_data


def split_edge_targets(
    data: Data,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate positive and negative decoder targets."""

    if data.edge_label is None:
        raise ValueError(
            "The split does not contain data.edge_label."
        )

    if data.edge_label_index is None:
        raise ValueError(
            "The split does not contain data.edge_label_index."
        )

    positive_mask = data.edge_label == 1
    negative_mask = data.edge_label == 0

    positive_edge_index = data.edge_label_index[
        :,
        positive_mask,
    ]

    negative_edge_index = data.edge_label_index[
        :,
        negative_mask,
    ]

    if positive_edge_index.size(1) == 0:
        raise ValueError(
            "The evaluation split contains no positive edges."
        )

    if negative_edge_index.size(1) == 0:
        raise ValueError(
            "The evaluation split contains no negative edges."
        )

    return positive_edge_index, negative_edge_index


def evaluate_link_reconstruction(
    model: GAE,
    data: Data,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate reconstruction on fixed positive and negative pairs."""

    model.eval()

    positive_edge_index, negative_edge_index = (
        split_edge_targets(data)
    )

    x = data.x.to(device)
    message_edge_index = data.edge_index.to(device)

    positive_edge_index = positive_edge_index.to(device)
    negative_edge_index = negative_edge_index.to(device)

    candidate_edge_index = data.edge_label_index.to(device)
    candidate_labels = data.edge_label.to(device)

    with torch.no_grad():
        # Build embeddings using only graph connectivity
        # permitted for this evaluation split.
        z = model.encode(
            x,
            message_edge_index,
        )

        # Use the fixed positive and negative pairs so the
        # metric remains comparable between epochs.
        reconstruction_loss = model.recon_loss(
            z,
            positive_edge_index,
            negative_edge_index,
        )

        candidate_probabilities = model.decode(
            z,
            candidate_edge_index,
        )

    labels_numpy = (
        candidate_labels
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    probabilities_numpy = (
        candidate_probabilities
        .detach()
        .cpu()
        .numpy()
    )

    roc_auc = roc_auc_score(
        labels_numpy,
        probabilities_numpy,
    )

    average_precision = average_precision_score(
        labels_numpy,
        probabilities_numpy,
    )

    positive_probabilities = candidate_probabilities[
        candidate_labels == 1
    ]

    negative_probabilities = candidate_probabilities[
        candidate_labels == 0
    ]

    return {
        "loss": float(reconstruction_loss.item()),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "positive_edges": int(positive_edge_index.size(1)),
        "negative_edges": int(negative_edge_index.size(1)),
        "mean_positive_probability": float(
            positive_probabilities.mean().item()
        ),
        "mean_negative_probability": float(
            negative_probabilities.mean().item()
        ),
        "mean_positive_anomaly_score": float(
            1.0 - positive_probabilities.mean().item()
        ),
    }


def sample_training_negatives(
    full_positive_edge_index: torch.Tensor,
    num_nodes: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample non-edges while excluding every known graph edge."""

    negative_edge_index = negative_sampling(
        # This is the complete prepared graph, including
        # relationships later assigned to validation/test.
        edge_index=full_positive_edge_index,

        num_nodes=num_nodes,
        num_neg_samples=num_samples,

        # The reference graph already contains both directions.
        # Sampling one orientation per negative pair is enough
        # because the inner-product decoder is symmetric.
        force_undirected=False,

        method="sparse",
    )

    if negative_edge_index.size(1) == 0:
        raise ValueError(
            "Negative sampling returned no candidate pairs."
        )

    return negative_edge_index.to(device)




def validate_edge_splits(
    train_data: Data,
    val_data: Data,
    test_data: Data,
) -> None:
    """Validate edge-level train, validation, and test objects."""

    split_objects = {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }

    expected_num_nodes = train_data.num_nodes
    expected_feature_shape = train_data.x.shape

    for split_name, split_data in split_objects.items():
        if split_data.num_nodes != expected_num_nodes:
            raise ValueError(
                f"{split_name} split changed the node count."
            )

        if split_data.x.shape != expected_feature_shape:
            raise ValueError(
                f"{split_name} split changed the feature shape."
            )

        if split_data.edge_index.ndim != 2:
            raise ValueError(
                f"{split_name}.edge_index must be rank 2."
            )

        if split_data.edge_index.size(0) != 2:
            raise ValueError(
                f"{split_name}.edge_index must have shape [2, E]."
            )

        if split_data.edge_label_index.ndim != 2:
            raise ValueError(
                f"{split_name}.edge_label_index must be rank 2."
            )

        if split_data.edge_label_index.size(0) != 2:
            raise ValueError(
                f"{split_name}.edge_label_index must have "
                "shape [2, candidate_edges]."
            )

        if (
            split_data.edge_label_index.size(1)
            != split_data.edge_label.numel()
        ):
            raise ValueError(
                f"{split_name} has a different number of "
                "candidate edges and edge labels."
            )

    train_unique_labels = torch.unique(
        train_data.edge_label
    )

    # With add_negative_train_samples=False, every training
    # decoder target should be an observed positive edge.
    if not torch.all(train_data.edge_label == 1):
        raise ValueError(
            "Training edge labels must all equal 1 for this "
            "self-supervised reconstruction baseline. "
            f"Observed label values: {train_unique_labels.tolist()}. "
            "Check for a pre-existing edge_label on the input graph."
        )

    train_positive_count = train_data.edge_label.numel()
    train_negative_count = 0

    if train_positive_count == 0:
        raise ValueError(
            "The training split contains no reconstruction targets."
        )

    for split_name, split_data in {
        "validation": val_data,
        "test": test_data,
    }.items():
        positive_count = int(
            (split_data.edge_label == 1).sum().item()
        )
        negative_count = int(
            (split_data.edge_label == 0).sum().item()
        )

        if positive_count == 0:
            raise ValueError(
                f"{split_name} contains no positive edges."
            )

        if negative_count == 0:
            raise ValueError(
                f"{split_name} contains no negative edges."
            )
        
def undirected_edge_ids(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Convert undirected node pairs into unique integer IDs."""

    source = edge_index[0].cpu()
    target = edge_index[1].cpu()

    lower_node = torch.minimum(source, target)
    higher_node = torch.maximum(source, target)

    # Pair (i, j) becomes one unique integer:
    # edge_id = i * num_nodes + j
    edge_ids = lower_node * num_nodes + higher_node

    return torch.unique(edge_ids)


def audit_split_leakage(
    val_data: Data,
    test_data: Data,
) -> None:
    """Ensure held-out positive edges are absent from their context graphs."""

    num_nodes = val_data.num_nodes

    # Positive validation relationships only.
    val_positive_mask = (
        val_data.edge_label == 1
    )
    val_positive_edges = (
        val_data.edge_label_index[:, val_positive_mask]
    )

    # Positive test relationships only.
    test_positive_mask = (
        test_data.edge_label == 1
    )
    test_positive_edges = (
        test_data.edge_label_index[:, test_positive_mask]
    )

    val_context_ids = undirected_edge_ids(
        val_data.edge_index,
        num_nodes,
    )
    val_target_ids = undirected_edge_ids(
        val_positive_edges,
        num_nodes,
    )

    test_context_ids = undirected_edge_ids(
        test_data.edge_index,
        num_nodes,
    )
    test_target_ids = undirected_edge_ids(
        test_positive_edges,
        num_nodes,
    )

    validation_leakage = torch.isin(
        val_target_ids,
        val_context_ids,
    ).any()

    test_leakage = torch.isin(
        test_target_ids,
        test_context_ids,
    ).any()

    if bool(validation_leakage):
        raise ValueError(
            "Validation leakage detected: at least one "
            "positive validation edge is visible to the encoder."
        )

    if bool(test_leakage):
        raise ValueError(
            "Test leakage detected: at least one positive "
            "test edge is visible to the encoder."
        )

    print("\nEdge leakage audit")
    print("------------------")
    print("Validation target leakage: none")
    print("Test target leakage: none")


def print_edge_split_summary(
    train_data: Data,
    val_data: Data,
    test_data: Data,
) -> None:
    """Print message-passing and reconstruction edge counts."""

    print("\nEdge split summary")
    print("------------------")

    for split_name, split_data in [
        ("Training", train_data),
        ("Validation", val_data),
        ("Test", test_data),
    ]:
        positive_count = int(
            (split_data.edge_label == 1).sum().item()
        )
        negative_count = int(
            (split_data.edge_label == 0).sum().item()
        )

        print(f"\n{split_name}")
        print(
            "  Message-passing edge entries:",
            f"{split_data.edge_index.size(1):,}",
        )
        print(
            "  Decoder candidate pairs:",
            f"{split_data.edge_label_index.size(1):,}",
        )
        print(
            "  Positive targets:",
            f"{positive_count:,}",
        )
        print(
            "  Negative targets:",
            f"{negative_count:,}",
        )


def load_graph(graph_path: Path) -> Data:
    """Load the saved homogeneous PyG graph on the CPU."""

    if not graph_path.exists():
        raise FileNotFoundError(
            f"LANL graph file does not exist: {graph_path}"
        )

    try:
        data = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        data = torch.load(
            graph_path,
            map_location="cpu",
        )

    if not isinstance(data, Data):
        raise TypeError(
            "Expected the graph file to contain a "
            f"torch_geometric.data.Data object, got {type(data)}."
        )

    return data


def print_graph_summary(data: Data) -> None:
    """Print important graph properties before edge splitting."""

    num_nodes = data.x.size(0)
    num_features = data.x.size(1)
    num_edge_entries = data.edge_index.size(1)

    source_nodes = data.edge_index[0]
    target_nodes = data.edge_index[1]

    self_loop_count = int(
        (source_nodes == target_nodes).sum().item()
    )

    # Count every node that appears as either a source or target.
    incident_node_ids = torch.cat(
        [source_nodes, target_nodes],
        dim=0,
    )

    incident_counts = torch.bincount(
        incident_node_ids,
        minlength=num_nodes,
    )

    isolated_node_count = int(
        (incident_counts == 0).sum().item()
    )

    print("\nLANL graph summary")
    print("------------------")
    print(f"Nodes: {num_nodes:,}")
    print(f"Node features: {num_features:,}")
    print(f"Directed edge entries: {num_edge_entries:,}")
    print(f"Node feature dtype: {data.x.dtype}")
    print(f"Edge-index dtype: {data.edge_index.dtype}")
    print(f"Undirected representation: {data.is_undirected()}")
    print(f"Self-loop entries: {self_loop_count:,}")
    print(f"Isolated nodes: {isolated_node_count:,}")

    if data.y is not None:
        print(f"Node-label shape: {tuple(data.y.shape)}")
    else:
        print("Node labels: not present")


class GCNEncoder(nn.Module):
    """Encode LANL graph nodes into latent vectors."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        latent_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        # First graph-convolution layer:
        # [N, input_channels] -> [N, hidden_channels]
        self.conv1 = GCNConv(
            in_channels=input_channels,
            out_channels=hidden_channels,
        )

        # Final graph-convolution layer:
        # [N, hidden_channels] -> [N, latent_channels]
        self.conv2 = GCNConv(
            in_channels=hidden_channels,
            out_channels=latent_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one latent embedding for every graph node."""

        # Each node combines its own transformed features
        # with normalized information from graph neighbors.
        h = self.conv1(
            x,
            edge_index,
        )

        # Shape:
        # h = [N, hidden_channels]
        h = F.relu(h)

        # Dropout is active only while model.training is True.
        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        # Produce the final latent node embeddings.
        z = self.conv2(
            h,
            edge_index,
        )

        # Shape:
        # z = [N, latent_channels]
        return z
    

def build_model(
    data: Data,
    config: dict[str, Any],
) -> GAE:
    """Build the PyG graph autoencoder."""

    input_channels = int(data.x.size(1))

    hidden_channels = int(
        config["model"]["hidden_channels"]
    )

    latent_channels = int(
        config["model"]["latent_channels"]
    )

    dropout = float(
        config["model"]["dropout"]
    )

    encoder = GCNEncoder(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        dropout=dropout,
    )

    # Since no decoder is supplied, PyG uses its
    # default inner-product edge decoder.
    model = GAE(encoder)
    # GAE wapper provides the following methods for training and evaluation:
    # model.encode(...)
    # model.decode(...)
    # model.recon_loss(...)
    # model.test(...)
    # Conceptually 
    # GCNEncoder owns trainable graph-convolution parameters
    # GAE coordinates encoder, decoder, and reconstruction loss
 
    return model 


def flatten_config(
    config: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested YAML fields into MLflow parameters."""

    flattened: dict[str, Any] = {}

    for key, value in config.items():
        parameter_name = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(value, dict):
            flattened.update(
                flatten_config(
                    value,
                    prefix=parameter_name,
                )
            )
        else:
            flattened[parameter_name] = value

    return flattened



def configure_mlflow(
    config: dict[str, Any],
) -> str:
    """Configure local MLflow tracking and return experiment ID."""

    tracking_uri = str(
        config["experiment_tracking"]["uri"]
    )

    # Ensure the parent directory for the SQLite database exists.
    sqlite_prefix = "sqlite:///"

    if tracking_uri.startswith(sqlite_prefix):
        database_path = Path(
            tracking_uri[len(sqlite_prefix):]
        )

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    artifact_root = Path(
        config["experiment_tracking"]["artifact_root"]
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = str(
        config["experiment_tracking"]["experiment_name"]
    )

    existing_experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if existing_experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=(
                artifact_root.resolve().as_uri()
            ),
        )
    else:
        experiment_id = existing_experiment.experiment_id

    return experiment_id



def compute_test_safe_embeddings(
    model: GAE,
    test_data: Data,
    device: torch.device,
) -> torch.Tensor:
    """Create embeddings without exposing held-out test edges."""

    model.eval()

    with torch.no_grad():
        z = model.encode(
            test_data.x.to(device),
            test_data.edge_index.to(device),
        )

    if not torch.isfinite(z).all():
        raise ValueError(
            "Final node embeddings contain non-finite values."
        )

    return z.detach().cpu()


def make_json_safe(value: Any) -> Any:
    """Convert common experiment objects into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def save_experiment_artifacts(
    model: GAE,
    embeddings: torch.Tensor,
    metrics_report: dict[str, Any],
    config: dict[str, Any],
    best_epoch: int,
) -> tuple[Path, Path, Path]:
    """Save checkpoint, metrics, and test-safe embeddings."""

    model_path = Path(
        config["output"]["model_path"]
    )

    metrics_path = Path(
        config["output"]["metrics_path"]
    )

    embeddings_path = Path(
        config["output"]["embeddings_path"]
    )

    for output_path in [
        model_path,
        metrics_path,
        embeddings_path,
    ]:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    cpu_model_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": cpu_model_state,
            "model_type": "graph_autoencoder",
            "encoder_type": "two_layer_gcn",
            "decoder_type": "symmetric_inner_product",
            "input_channels": int(
                metrics_report["graph"]["input_channels"]
            ),
            "hidden_channels": int(
                config["model"]["hidden_channels"]
            ),
            "latent_channels": int(
                config["model"]["latent_channels"]
            ),
            "dropout": float(
                config["model"]["dropout"]
            ),
            "best_epoch": best_epoch,
            "config": config,
        },
        model_path,
    )

    torch.save(
        {
            "embeddings": embeddings,
            "shape": list(embeddings.shape),
            "context": (
                "Encoded with test_data.edge_index: "
                "training and validation connectivity available; "
                "test-positive relationships excluded."
            ),
            "decoder_type": "symmetric_inner_product",
        },
        embeddings_path,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(metrics_report),
            file,
            indent=2,
        )

    return (
        model_path,
        metrics_path,
        embeddings_path,
    )


def build_metrics_report(
    config: dict[str, Any],
    prepared_data: Data,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    history: list[dict[str, float | int]],
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    device: torch.device,
) -> dict[str, Any]:
    """Build the complete experiment report."""

    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": {
            "family": "graph_autoencoder",
            "encoder": "two_layer_gcn",
            "decoder": "symmetric_inner_product",
            "training_objective": (
                "self_supervised_link_reconstruction"
            ),
        },

        "graph": {
            "nodes": int(prepared_data.num_nodes),
            "input_channels": int(
                prepared_data.x.size(1)
            ),
            "prepared_directed_edge_entries": int(
                prepared_data.edge_index.size(1)
            ),
            "prepared_undirected_relationships": int(
                prepared_data.edge_index.size(1) // 2
            ),
            "source_graph_was_symmetrized": True,
        },

        "split": {
            "train_positive_edges": int(
                train_data.edge_label_index.size(1)
            ),
            "validation_candidate_edges": int(
                val_data.edge_label_index.size(1)
            ),
            "test_candidate_edges": int(
                test_data.edge_label_index.size(1)
            ),
            "val_ratio": float(
                config["split"]["val_ratio"]
            ),
            "test_ratio": float(
                config["split"]["test_ratio"]
            ),
            "negative_sampling_ratio": float(
                config["split"][
                    "negative_sampling_ratio"
                ]
            ),
            "seed": int(config["split"]["seed"]),
        },

        "training": {
            "device": str(device),
            "epochs_requested": int(
                config["training"]["epochs"]
            ),
            "best_epoch": best_epoch,
            "checkpoint_selection_metric": (
                "validation_average_precision"
            ),
            "history": history,
        },

        "best_validation": best_validation_metrics,
        "test": test_metrics,

        "interpretation": {
            "positive_target": (
                "Held-out structural relationship that exists."
            ),
            "negative_target": (
                "Sampled node pair with no known relationship."
            ),
            "anomaly_score": (
                "For an observed candidate edge, "
                "1 - reconstructed edge probability."
            ),
            "important_limitation": (
                "ROC-AUC and average precision measure "
                "link reconstruction, not malicious-event "
                "classification."
            ),
        },
    }


def train_graph_autoencoder(
    model: GAE,
    train_data: Data,
    val_data: Data,
    full_positive_edge_index: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    list[dict[str, float | int]],
    int,
    dict[str, float | int],
]:
    """Train the GAE and restore the best validation checkpoint."""

    epochs = int(config["training"]["epochs"])
    learning_rate = float(
        config["training"]["learning_rate"]
    )
    weight_decay = float(
        config["training"]["weight_decay"]
    )

    log_every = int(
        config["training"].get(
            "log_every_n_epochs",
            10,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_x = train_data.x.to(device)
    train_message_edges = train_data.edge_index.to(device)

    # Training edge_label_index contains observed positive
    # relationships only because fixed negatives were disabled.
    train_positive_edges = (
        train_data.edge_label_index.to(device)
    )

    best_epoch = 0
    best_validation_score = float("-inf")
    best_validation_metrics: dict[str, float | int] = {}
    best_model_state: dict[str, torch.Tensor] | None = None

    history: list[dict[str, float | int]] = []

    print("\nTraining")
    print("--------")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # [N, F_in] -> [N, latent_channels]
        z = model.encode(
            train_x,
            train_message_edges,
        )

        # Sample a fresh negative set during every epoch.
        train_negative_edges = sample_training_negatives(
            full_positive_edge_index=full_positive_edge_index,
            num_nodes=train_data.num_nodes,
            num_samples=train_positive_edges.size(1),
            device=device,
        )

        # Positive edges should approach probability 1.
        # Negative edges should approach probability 0.
        train_loss = model.recon_loss(
            z,
            train_positive_edges,
            train_negative_edges,
        )

        if not torch.isfinite(train_loss):
            raise ValueError(
                f"Training loss became non-finite at epoch {epoch}."
            )

        train_loss.backward()
        optimizer.step()

        validation_metrics = evaluate_link_reconstruction(
            model=model,
            data=val_data,
            device=device,
        )

        epoch_result: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "val_loss": float(
                validation_metrics["loss"]
            ),
            "val_roc_auc": float(
                validation_metrics["roc_auc"]
            ),
            "val_average_precision": float(
                validation_metrics["average_precision"]
            ),
        }

        history.append(epoch_result)

        # Average precision is the checkpoint-selection metric.
        current_validation_score = float(
            validation_metrics["average_precision"]
        )

        if current_validation_score > best_validation_score:
            best_validation_score = current_validation_score
            best_epoch = epoch
            best_validation_metrics = dict(
                validation_metrics
            )

            # Store the best checkpoint on CPU so it does not
            # consume additional accelerator memory.
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor
                in model.state_dict().items()
            }

        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss.item()),
                    "val_loss": float(
                        validation_metrics["loss"]
                    ),
                    "val_roc_auc": float(
                        validation_metrics["roc_auc"]
                    ),
                    "val_average_precision": float(
                        validation_metrics[
                            "average_precision"
                        ]
                    ),
                },
                step=epoch,
            )

        should_print = (
            epoch == 1
            or epoch == epochs
            or epoch % log_every == 0
        )

        if should_print:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss={train_loss.item():.4f} | "
                f"val loss={validation_metrics['loss']:.4f} | "
                f"val ROC-AUC="
                f"{validation_metrics['roc_auc']:.4f} | "
                f"val AP="
                f"{validation_metrics['average_precision']:.4f}"
            )

    if best_model_state is None:
        raise RuntimeError(
            "Training completed without a valid checkpoint."
        )

    # Restore the parameters from the epoch with the best
    # validation average precision.
    model.load_state_dict(best_model_state)

    print("\nBest validation checkpoint")
    print("--------------------------")
    print(f"Epoch: {best_epoch}")
    print(
        "Validation average precision:",
        f"{best_validation_metrics['average_precision']:.6f}",
    )
    print(
        "Validation ROC-AUC:",
        f"{best_validation_metrics['roc_auc']:.6f}",
    )

    return (
        history,
        best_epoch,
        best_validation_metrics,
    )


def print_model_summary(
    model: GAE,
    data: Data,
    config: dict[str, Any],
) -> None:
    """Print model architecture and tensor dimensions."""

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nGraph autoencoder")
    print("-----------------")
    print(model)
    print(f"Input channels: {data.x.size(1):,}")
    print(
        "Hidden channels:",
        f"{int(config['model']['hidden_channels']):,}",
    )
    print(
        "Latent channels:",
        f"{int(config['model']['latent_channels']):,}",
    )
    print(
        "Dropout:",
        float(config["model"]["dropout"]),
    )
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )
    print("Decoder: symmetric inner product")



def run_forward_smoke_test(
    model: GAE,
    train_data: Data,
    latent_channels: int,
    device: torch.device,
) -> None:
    """Verify encoder and decoder tensor shapes before training."""

    model.eval()

    with torch.no_grad():
        # Produce one embedding for every node using only
        # training message-passing connectivity.
        z = model.encode(
            train_data.x.to(device),
            train_data.edge_index.to(device),
        )

        expected_embedding_shape = (
            train_data.num_nodes,
            latent_channels,
        )

        if tuple(z.shape) != expected_embedding_shape:
            raise ValueError(
                "Unexpected embedding shape. "
                f"Expected {expected_embedding_shape}, "
                f"received {tuple(z.shape)}."
            )

        if not torch.isfinite(z).all():
            raise ValueError(
                "Encoder produced NaN or infinite embeddings."
            )

        positive_edge_index = (
            train_data.edge_label_index.to(device)
        )

        # Use only a small candidate subset for the smoke test.
        sample_size = min(
            32,
            positive_edge_index.size(1),
        )

        sample_edges = positive_edge_index[
            :,
            :sample_size,
        ]

        edge_probabilities = model.decode(
            z,
            sample_edges,
        )

        expected_probability_shape = (
            sample_size,
        )

        if tuple(edge_probabilities.shape) != (
            expected_probability_shape
        ):
            raise ValueError(
                "Unexpected decoder output shape. "
                f"Expected {expected_probability_shape}, "
                f"received {tuple(edge_probabilities.shape)}."
            )

        if not torch.isfinite(edge_probabilities).all():
            raise ValueError(
                "Decoder produced NaN or infinite probabilities."
            )

        if bool(
            (
                (edge_probabilities < 0.0)
                | (edge_probabilities > 1.0)
            ).any()
        ):
            raise ValueError(
                "Decoded edge probabilities must be in [0, 1]."
            )

    print("\nForward-pass smoke test")
    print("-----------------------")
    print(f"Embedding shape: {tuple(z.shape)}")
    print(
        "Decoded sample-edge shape:",
        tuple(edge_probabilities.shape),
    )
    print(
        "Minimum sample probability:",
        f"{edge_probabilities.min().item():.6f}",
    )
    print(
        "Maximum sample probability:",
        f"{edge_probabilities.max().item():.6f}",
    )
    print("Forward-pass validation: passed")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    seed = int(config["split"]["seed"])
    set_random_seeds(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Training device: {device}")

    graph_path = Path(
        config["input"]["graph_path"]
    )

    data = load_graph(graph_path)

    expect_undirected = bool(
        config["split"]["is_undirected"]
    )

    validate_graph(
        data=data,
        expect_undirected=expect_undirected,
    )

    print("Configuration loaded successfully.")
    print(f"Configuration path: {args.config}")
    print(f"Graph path: {graph_path}")
    print(f"Random seed: {seed}")

    print_graph_summary(data)


    prepared_data = prepare_link_prediction_graph(
        data=data,
        make_undirected=expect_undirected,
    )
    train_data, val_data, test_data = create_edge_splits(
        data=prepared_data,
        config=config,
    )

    validate_edge_splits(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    audit_split_leakage(
        val_data=val_data,
        test_data=test_data,
    )

    print_edge_split_summary(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    experiment_id = configure_mlflow(config)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="lanl-graph-autoencoder",
    ):
        mlflow.log_params(
            flatten_config(config)
        )

        mlflow.log_params(
            {
                "graph.nodes": int(
                    prepared_data.num_nodes
                ),
                "graph.input_channels": int(
                    prepared_data.x.size(1)
                ),
                "graph.prepared_edge_entries": int(
                    prepared_data.edge_index.size(1)
                ),
                "device": str(device),
            }
        )

        model = build_model(
            data=train_data,
            config=config,
        )

        model = model.to(device)

        print_model_summary(
            model=model,
            data=train_data,
            config=config,
        )

        run_forward_smoke_test(
            model=model,
            train_data=train_data,
            latent_channels=int(
                config["model"]["latent_channels"]
            ),
            device=device,
        )

        (
            history,
            best_epoch,
            best_validation_metrics,
        ) = train_graph_autoencoder(
            model=model,
            train_data=train_data,
            val_data=val_data,

            # Use the complete prepared graph so held-out
            # true edges cannot become training negatives.
            full_positive_edge_index=(
                prepared_data.edge_index
            ),

            config=config,
            device=device,
        )

        test_metrics = evaluate_link_reconstruction(
            model=model,
            data=test_data,
            device=device,
        )

        print("\nHeld-out test results")
        print("---------------------")
        print(
            "Test loss:",
            f"{test_metrics['loss']:.6f}",
        )
        print(
            "Test ROC-AUC:",
            f"{test_metrics['roc_auc']:.6f}",
        )
        print(
            "Test average precision:",
            f"{test_metrics['average_precision']:.6f}",
        )
        print(
            "Mean positive-edge probability:",
            f"{test_metrics['mean_positive_probability']:.6f}",
        )
        print(
            "Mean negative-edge probability:",
            f"{test_metrics['mean_negative_probability']:.6f}",
        )

        embeddings = compute_test_safe_embeddings(
            model=model,
            test_data=test_data,
            device=device,
        )

        metrics_report = build_metrics_report(
            config=config,
            prepared_data=prepared_data,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            history=history,
            best_epoch=best_epoch,
            best_validation_metrics=(
                best_validation_metrics
            ),
            test_metrics=test_metrics,
            device=device,
        )

        (
            model_path,
            metrics_path,
            embeddings_path,
        ) = save_experiment_artifacts(
            model=model,
            embeddings=embeddings,
            metrics_report=metrics_report,
            config=config,
            best_epoch=best_epoch,
        )

        mlflow.log_metrics(
            {
                "best_epoch": float(best_epoch),
                "best_val_roc_auc": float(
                    best_validation_metrics["roc_auc"]
                ),
                "best_val_average_precision": float(
                    best_validation_metrics[
                        "average_precision"
                    ]
                ),
                "test_loss": float(
                    test_metrics["loss"]
                ),
                "test_roc_auc": float(
                    test_metrics["roc_auc"]
                ),
                "test_average_precision": float(
                    test_metrics["average_precision"]
                ),
                "test_mean_positive_probability": float(
                    test_metrics[
                        "mean_positive_probability"
                    ]
                ),
                "test_mean_negative_probability": float(
                    test_metrics[
                        "mean_negative_probability"
                    ]
                ),
            }
        )

        mlflow.log_artifact(
            str(args.config),
            artifact_path="config",
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="reports",
        )

        mlflow.log_artifact(
            str(embeddings_path),
            artifact_path="embeddings",
        )

        print("\nSaved artifacts")
        print("---------------")
        print(f"Model: {model_path}")
        print(f"Metrics: {metrics_path}")
        print(f"Embeddings: {embeddings_path}")

        

if __name__ == "__main__":
    main()
