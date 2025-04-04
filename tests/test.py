# Some general parameters we need
import time

import client
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from flwr_contributions.flwr_contributions import flwr_contribution_strategy

init_model = client.get_model()
initial_parameters = ndarrays_to_parameters(client.ndarrays_from_model(init_model))

del init_model
torch.cuda.empty_cache()

import flwr as fl

client_resources: dict = {
    "num_cpus": 1,
    "num_gpus": 1,
}

FedContrib = flwr_contribution_strategy(FedAvg)

starttime = time.time()
hist = fl.simulation.start_simulation(
    client_fn=client.client_fn,
    num_clients=4,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=FedContrib(
        initial_parameters=initial_parameters,
        evaluate_fn=client.evaluate_fn,
        on_fit_config_fn=client.fit_config,
    ),
    client_resources=client_resources,
)
