import logging
from typing import List, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import SizePartitioner
from flwr_datasets.visualization import plot_label_distributions


def generate_datasplit(num_clients: int, dataset: str, total_size: int = None):
    """Create a uniform federated split of a dataset.

    If total_size is not provided, partitions are created with equal fractional sizes.
    """
    if total_size is None:
        partition_sizes = [1.0 / num_clients] * num_clients
    else:
        partition_sizes = [int(total_size / num_clients)] * num_clients
    fds = FederatedDataset(
        dataset=dataset,
        partitioners={"train": SizePartitioner(partition_sizes)},
    )
    logging.info("Dataset successfully split.")
    return fds


def flip_indices(
    dataset: Dataset, target_name: str = "label", perc_malicious: float = 0.35
):
    """Corrupt a fraction of labels by flipping them to a random other class."""
    old_labels = np.array(dataset[target_name])
    size = len(old_labels)
    targets = np.unique(old_labels)

    malicious_indices = np.random.choice(
        size, size=int(perc_malicious * size), replace=False
    )
    for i in malicious_indices:
        old_labels[i] = np.random.choice(targets[targets != old_labels[i]])

    dataset = (
        dataset.remove_columns(target_name)
        .add_column(target_name, old_labels.tolist())
        .cast(dataset.features)
    )
    return dataset


def ndarrays_from_model(model: torch.nn.Module) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.Module, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def client_threshold(num_clients: int, min_clients: int, current_round: int) -> int:
    """Dynamic client participation threshold via sigmoid schedule."""
    def sigm(x):
        return 1 / (1 + np.exp(-x))

    return min_clients + int(num_clients - num_clients * sigm(current_round / 5))


def plot_distributions(partitioner, labelname: str, dsname: str, save_dir: str = "./plots"):
    """Plot and save per-partition label distributions."""
    fig, ax, df = plot_label_distributions(
        partitioner,
        label_name=labelname,
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
    )
    plt.savefig(f"{save_dir}/label_distribution_{dsname}.pdf")
