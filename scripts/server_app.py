"""flwr-contributions: Server app with Shapley-value contribution measurement."""

import importlib
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from flwr.common import Context, FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flwr_shapley.federated_shapley import MultiRoundReconstructor, OneRoundReconstructor
from shapley.kernel_samplers import KendallSampler, MallowsSampler, SpearmanSampler
from shapley.probabilistic_samplers import (
    AntitheticMonteCarloSampler,
    FullSampler,
    MonteCarloSampler,
    Sampler,
    StratifiedSampler,
)
from task.data_utils import ndarrays_from_model, ndarrays_to_model

logger = logging.getLogger(__name__)


def create_contribution_strategy(
    parent_strategy=FedAvg,
    sampler: Sampler = MonteCarloSampler,
    method: str = "one-round",
    num_rounds: int = 1,
):
    """Create a strategy subclass that measures client contributions via Shapley values.

    Parameters
    ----------
    parent_strategy : Strategy class
        The base strategy to wrap (e.g., FedAvg).
    sampler : Sampler
        Sampler instance for generating permutations.
    method : str
        "one-round" accumulates gradients, evaluates at the end.
        "multi-round" evaluates subsets independently each round.
    num_rounds : int
        Total number of rounds (needed for one-round to know when to evaluate).
    """

    class ContributionStrategy(parent_strategy):
        def __init__(
            self,
            initial_parameters: Parameters = None,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Callable = None,
        ) -> None:
            super().__init__(
                initial_parameters=initial_parameters,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config_fn,
            )
            self.sampler = sampler
            self.method = method
            self.reconstructor = None
            # Keep our own copy since FedAvg sets self.initial_parameters = None
            self._current_parameters = parameters_to_ndarrays(initial_parameters) if initial_parameters else None
            if method == "one-round":
                self.reconstructor = OneRoundReconstructor(
                    sampler=sampler,
                    evaluate_fn=evaluate_fn,
                    num_rounds=num_rounds,
                )
            elif method == "multi-round":
                self.reconstructor = MultiRoundReconstructor(
                    sampler=sampler,
                    evaluate_fn=evaluate_fn,
                    num_rounds=num_rounds,
                    aggregate_fit=super().aggregate_fit,
                )

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            if not results:
                return None, {}
            if not self.accept_failures and failures:
                return None, {}

            if self.method == "one-round":
                # Get current global parameters to compute gradients against
                current_params = self._current_parameters
                shapley_result = self.reconstructor.one_round(
                    server_round, results, current_params,
                )
                # Print permutations on first round
                if server_round == 1:
                    print(f"[shapley] Permutations: {self.reconstructor.permutations}")
                # Print Shapley values on final round
                if shapley_result is not None:
                    print(f"[shapley] Final Shapley values: {shapley_result}")

            elif self.method == "multi-round":
                shapley_result = self.reconstructor.on_round(
                    server_round, results, failures,
                )
                if shapley_result is not None:
                    print(f"[shapley] Final Shapley value history: {shapley_result}")

            # Perform normal aggregation
            aggregated = super().aggregate_fit(server_round, results, failures)

            # Update current parameters for next round (used as baseline for gradients)
            if aggregated[0] is not None and self.method == "one-round":
                self._current_parameters = parameters_to_ndarrays(aggregated[0])

            return aggregated

    return ContributionStrategy


def server_fn(context: Context) -> ServerAppComponents:
    run_config = context.run_config
    num_rounds: int = int(run_config["num-server-rounds"])
    num_clients: int = int(run_config["num-clients"])
    experiment: str = run_config["experiment"]
    use_contributions: bool = bool(run_config.get("use-contributions", False))
    contribution_method: str = str(run_config.get("contribution-method", "one-round"))
    sampler_type: str = str(run_config.get("sampler-type", "monte-carlo"))
    num_samples: int = int(run_config.get("num-samples", 2))

    SAMPLERS = {
        "monte-carlo": MonteCarloSampler,
        "antithetic": AntitheticMonteCarloSampler,
        #"stratified": StratifiedSampler,
        "full": FullSampler,
        "kendall": KendallSampler,
        "mallows": MallowsSampler,
        "spearman": SpearmanSampler,
    }
    SamplerClass = SAMPLERS.get(sampler_type, MonteCarloSampler)

    # Dynamically import the task module
    mltask = importlib.import_module(f"task.{experiment}")

    # Get initial parameters
    init_model = mltask.get_model()
    initial_parameters = ndarrays_to_parameters(ndarrays_from_model(init_model))
    del init_model
    torch.cuda.empty_cache()

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate global model on central test data."""
        model = mltask.get_model()
        ndarrays_to_model(model, parameters_ndarrays)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        testloader = mltask.load_global_test_data()
        loss, accuracy = mltask.test(model, testloader, device)
        del model
        torch.cuda.empty_cache()
        return loss, {"accuracy": accuracy}

    if use_contributions:
        StrategyClass = create_contribution_strategy(
            parent_strategy=FedAvg,
            sampler=SamplerClass(samplesize=num_samples, seed=42),
            method=contribution_method,
            num_rounds=num_rounds,
        )
    else:
        StrategyClass = FedAvg

    strategy = StrategyClass(
        fraction_fit=float(run_config.get("fraction-fit", 1.0)),
        fraction_evaluate=float(run_config.get("fraction-evaluate", 0.0)),
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
