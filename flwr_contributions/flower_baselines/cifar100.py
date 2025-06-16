# Some general parameters we need
from typing import List, OrderedDict, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
import yaml
from datasets import Dataset
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    LinearPartitioner,
    SizePartitioner,
    SquarePartitioner,
)
from torch.utils.data import DataLoader

from flower_baselines.CUDA_VisionDataSet import CUDA_VisionDataSet


# Loading the model (Called when initializing FlowerClient and when testing)
def get_model() -> torch.nn.Module:
    return timm.create_model("resnet18").cuda()


transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)

transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)


def size_based_split(num_clients, num_high_quality):
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
    return FederatedDataset(
        dataset="uoft-cs/cifar100",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def dirichlet_based_split(num_clients, alpha):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="fine_label",
        alpha=alpha,
        min_partition_size=30,
    )
    return FederatedDataset(
        dataset="uoft-cs/cifar100",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


with open("./config/cifar100split_cfg.yaml", "r") as f:
    split_config = yaml.safe_load(f)

if split_config["split"] == "size":
    fds = size_based_split(split_config["num_clients"], split_config["num_quality"])

SIZES = [
    len(fds.load_partition(partition_id=partition_id))
    for partition_id in range(split_config["num_clients"])
]


def load_data(partition_id: int) -> DataLoader:
    dataset = fds.load_partition(partition_id=partition_id)
    trainloader = DataLoader(
        CUDA_VisionDataSet(
            dataset,
            transform=transform_train,
            image_name="img",
            target_name="fine_label",
        ),
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )
    return trainloader


# Load the (global) test dataset
def load_global_test_data() -> DataLoader:
    testset = fds.load_split("test")
    testloader = DataLoader(
        CUDA_VisionDataSet(
            testset,
            transform=transform_test,
            image_name="img",
            target_name="fine_label",
        ),
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )
    return testloader


# Train and test on a trainloader and testloader
def train(model: nn.Module, trainloader: DataLoader, **kwargs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    current_round = kwargs["ins"].config["current_round"]
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-3 * 0.99**current_round
    )
    for _ in range(1):
        for _, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test(
    model: nn.Module,
    testloader: DataLoader,
) -> Tuple[float, float]:
    """Validate the model on the entire test set."""
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


# All training calls, to be sent to the server and clients


from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg


def evaluate_fn(server_round, weights_aggregated, dict, **kwargs):
    model = get_model()
    ndarrays_to_model(model, weights_aggregated)
    loss, accuracy = test(model, load_global_test_data())
    del model
    torch.cuda.empty_cache()
    return -loss, {"accuracy": accuracy}


def get_initial_parameters():
    init_model = get_model()
    initial_parameters = ndarrays_to_parameters(ndarrays_from_model(init_model))
    del init_model
    torch.cuda.empty_cache()
    return initial_parameters


client_resources: dict = {
    "num_cpus": 1,
    "num_gpus": 0.1,
}
