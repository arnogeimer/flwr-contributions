import logging
from typing import List, OrderedDict, Tuple

import client
import data
import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client
from flwr.common import (Code, Context, FitIns, FitRes, Status,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from torch.utils.data import DataLoader


def get_model() -> torch.nn.Module:
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", weights=None, verbose=False
    ).cuda()


def train(model: nn.Module, trainloader: DataLoader, **kwargs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    for _ in range(5):
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


def ndarrays_from_model(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


trainloaders, testloader = data.split_data(4)


def load_data(partition_id: int) -> DataLoader:
    trainloader = trainloaders[partition_id]
    return trainloader


def load_global_test_data() -> DataLoader:
    return testloader


class FlowerClient(Client):
    def __init__(
        self,
        client_id: int,
        model=None,
    ) -> None:
        self.client_id: int = client_id
        self.model = model
        self.num_examples: int = None

    def fit(self, ins: FitIns) -> FitRes:

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        ndarrays_to_model(self.model, ndarrays_original)
        trainloader = load_data(
            self.client_id,
        )
        # Due to loading data onto the GPU, we
        if self.num_examples is None:
            self.num_examples = len(trainloader)
        train(
            model=self.model,
            trainloader=trainloader,
            ins=ins,
        )
        del trainloader
        torch.cuda.empty_cache()
        ndarrays_updated = ndarrays_from_model(self.model)

        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.num_examples,
            metrics={"client_id": self.client_id},
        )


def client_fn(context: Context) -> client.FlowerClient:
    partition_id = int(context.node_config["partition-id"])
    model = get_model()
    return FlowerClient(
        client_id=partition_id,
        model=model,
    ).to_client()


def evaluate_fn(server_round, weights_aggregated, dict, **kwargs):
    model = get_model()
    ndarrays_to_model(model, weights_aggregated)
    loss, accuracy = test(model, load_global_test_data())
    del model
    torch.cuda.empty_cache()
    return loss, {"accuracy": accuracy}


def fit_config(server_round: int):
    """Generate training configuration for each round."""
    config = {
        "current_round": server_round,
    }
    return config
