from typing import List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    PathologicalPartitioner,
    SizePartitioner,
)
from flwr_datasets.visualization import plot_label_distributions
from torch.utils.data import DataLoader


def get_model() -> torch.nn.Module:
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "Wav2Vec2Model", weights=None, verbose=False
    ).cuda()


def size_based_split(num_clients, num_high_quality=0):
    size_min, size_max = [
        max(256, 50000 / (num_clients * 3)),
        int(50000 / (num_clients * 2)),
    ]
    sizes = [np.random.randint(size_min, size_max) for _ in range(num_clients)]
    remaining_points = 50000 - np.sum(sizes[num_high_quality:])
    for i in range(num_high_quality):
        sizes[i] = int(remaining_points / num_high_quality)
    partitioner = SizePartitioner(
        partition_sizes=sizes,
    )
    fds = FederatedDataset(
        dataset="google/speech_commands",
        partitioners={"train": partitioner},
        trust_remote_code=True,
        subset="v0.01",
    )
    return fds


def pathological_split(num_clients):
    partitioner = PathologicalPartitioner(
        num_partitions=num_clients,
        num_classes_per_partition=5,
        partition_by="speaker_id",
    )

    fds = FederatedDataset(
        dataset="google/speech_commands",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )
    return fds
