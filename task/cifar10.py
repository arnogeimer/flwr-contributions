import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import tqdm
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    SizePartitioner,
)
from torch.utils.data import DataLoader


def get_model() -> torch.nn.Module:
    return timm.create_model("resnet18").cuda()


def _to_tensor(batch):
    """Convert PIL images to tensors."""
    batch["img"] = [torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in batch["img"]]
    return batch


def size_based_split(num_clients):
    partitioner = SizePartitioner(
        partition_sizes=[int(50000 / num_clients) for _ in range(num_clients)],
    )
    return partitioner


def dirichlet_based_split(num_clients, alpha):
    return DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=30,
    )


_test_fds = None


def _get_test_fds():
    global _test_fds
    if _test_fds is None:
        _test_fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": size_based_split(2)},
        )
    return _test_fds


def load_data(
    partition_id: int,
    num_clients: int = 36,
    num_workers: int = 4,
    batch_size: int = 1024,
    split_method: str = "uniform",
    dirichlet_alpha: float = 0.5,
) -> DataLoader:
    if split_method == "dirichlet":
        partitioner = dirichlet_based_split(num_clients, dirichlet_alpha)
    else:
        partitioner = size_based_split(num_clients)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    dataset = fds.load_partition(partition_id=partition_id)

    dataset = dataset.with_transform(_to_tensor)

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return trainloader


def load_global_test_data(batch_size: int = 64) -> DataLoader:
    testset = _get_test_fds().load_split("test")

    testset = testset.with_transform(_to_tensor)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return testloader


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 5,
    device: str = "cuda",
    lr: float = 1e-3,
    **kwargs,
):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting training")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    for _ in tqdm.trange(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test(
    model: nn.Module,
    testloader: DataLoader,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Validate the model on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    if len(testloader) == 0:
        return np.inf, 0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    del (testloader, model)
    torch.cuda.empty_cache()
    return loss, accuracy


def get_initial_state_dict():
    init_model = get_model()
    initial_state_dict = init_model.state_dict()
    del init_model
    torch.cuda.empty_cache()
    return initial_state_dict
