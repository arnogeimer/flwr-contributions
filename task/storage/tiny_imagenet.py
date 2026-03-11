from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    SizePartitioner,
)
from torch.utils.data import DataLoader

from task.cuda_vision import CUDA_VisionDataSet


def get_model() -> torch.nn.Module:
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "densenet121", weights=None, verbose=False
    ).cuda()


transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def size_based_split(num_clients, num_high_quality=0):
    size_min, size_max = [
        max(256, 100000 / (num_clients * 3)),
        int(100000 / (num_clients * 2)),
    ]
    sizes = [np.random.randint(size_min, size_max) for _ in range(num_clients)]
    remaining_points = 100000 - np.sum(sizes[num_high_quality:])
    for i in range(num_high_quality):
        sizes[i] = int(remaining_points / num_high_quality)
    partitioner = SizePartitioner(
        partition_sizes=sizes,
    )
    return FederatedDataset(
        dataset="zh-plus/tiny-imagenet",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def dirichlet_based_split(num_clients, alpha):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=30,
    )
    return FederatedDataset(
        dataset="zh-plus/tiny-imagenet",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def load_data(partition_id: int, fds: FederatedDataset, batch_size: int = 64) -> DataLoader:
    dataset = fds.load_partition(partition_id=partition_id)

    def apply_transforms(batch):
        batch["image"] = [
            torchvision.transforms.ToTensor()(transform_train(img))
            for img in batch["image"]
        ]
        return batch

    dataset = dataset.with_transform(apply_transforms)

    trainloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return trainloader


def load_data_onto_gpu(partition_id: int, fds: FederatedDataset, batch_size: int = 64) -> DataLoader:
    dataset = fds.load_partition(partition_id=partition_id)
    trainloader = DataLoader(
        CUDA_VisionDataSet(
            dataset,
            transform=transform_train,
            image_name="image",
            target_name="label",
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return trainloader


def load_global_test_data(fds: FederatedDataset, batch_size: int = 64) -> DataLoader:
    testset = fds.load_split("valid")

    def apply_test_transform(batch):
        batch["image"] = [
            torchvision.transforms.ToTensor()(transform_test(img))
            for img in batch["image"]
        ]
        return batch

    testset = testset.with_transform(apply_test_transform)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return testloader


def load_global_test_data_onto_gpu(fds: FederatedDataset, batch_size: int = 64) -> DataLoader:
    testset = fds.load_split("valid")
    testloader = DataLoader(
        CUDA_VisionDataSet(
            testset,
            transform=transform_test,
            image_name="image",
            target_name="label",
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return testloader


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 5,
    device: str = "cuda",
    **kwargs,
):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting training")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    for _ in tqdm.trange(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def train_on_gpu(model: nn.Module, trainloader: DataLoader, epochs: int = 5, current_round: int = 1, **kwargs):
    """Train the model on the training set (GPU-resident data)."""
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting training")
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-3 * 0.99**current_round
    )
    for _ in tqdm.trange(epochs):
        for _, (images, labels) in enumerate(trainloader):
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
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    del (testloader, model)
    torch.cuda.empty_cache()
    return loss, accuracy


def test_on_gpu(
    model: nn.Module,
    testloader: DataLoader,
) -> Tuple[float, float]:
    """Validate the model on the entire test set (GPU-resident data)."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    if len(testloader) == 0:
        return np.inf, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0], data[1]
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    del (testloader, model)
    torch.cuda.empty_cache()
    return loss, accuracy


def ndarrays_from_model(model: torch.nn.ModuleList) -> NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: NDArrays):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_initial_state_dict():
    init_model = get_model()
    initial_state_dict = init_model.state_dict()
    del init_model
    torch.cuda.empty_cache()
    return initial_state_dict
