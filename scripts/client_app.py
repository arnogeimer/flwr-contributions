"""flwr-contributions: Client app using modern Flower deployment."""

import importlib

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

class FlowerClient(NumPyClient):
    def __init__(self, mltask, partition_id, num_clients, batch_size, lr, epochs, device, split_method, dirichlet_alpha):
        self.mltask = mltask
        self.partition_id = partition_id
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.split_method = split_method
        self.dirichlet_alpha = dirichlet_alpha
        self.model = mltask.get_model()

    def get_parameters(self, config):
        from task.data_utils import ndarrays_from_model
        return ndarrays_from_model(self.model)

    def fit(self, parameters, config):
        from task.data_utils import ndarrays_to_model, ndarrays_from_model
        ndarrays_to_model(self.model, parameters)
        trainloader = self.mltask.load_data(
            self.partition_id,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            split_method=self.split_method,
            dirichlet_alpha=self.dirichlet_alpha,
        )
        self.mltask.train(
            self.model,
            trainloader,
            epochs=self.epochs,
            device=self.device,
            lr=self.lr,
        )
        return ndarrays_from_model(self.model), len(trainloader.dataset), {"client_id": self.partition_id}

    def evaluate(self, parameters, config):
        from task.data_utils import ndarrays_to_model
        ndarrays_to_model(self.model, parameters)
        testloader = self.mltask.load_global_test_data(batch_size=self.batch_size)
        loss, accuracy = self.mltask.test(self.model, testloader, self.device)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    run_config = context.run_config
    partition_id = int(context.node_config["partition-id"])
    experiment = run_config["experiment"]
    num_clients = int(run_config["num-clients"])
    batch_size = int(run_config.get("batch-size", 1024))
    lr = float(run_config.get("learning-rate", 1e-3))
    epochs = int(run_config.get("local-epochs", 1))
    split_method = str(run_config.get("split-method", "uniform"))
    dirichlet_alpha = float(run_config.get("dirichlet-alpha", 0.5))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mltask = importlib.import_module(f"task.{experiment}")

    return FlowerClient(
        mltask=mltask,
        partition_id=partition_id,
        num_clients=num_clients,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device,
        split_method=split_method,
        dirichlet_alpha=dirichlet_alpha,
    ).to_client()


app = ClientApp(client_fn=client_fn)
