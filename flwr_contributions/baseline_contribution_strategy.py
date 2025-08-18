import flwr_contributions as flcon
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)
import flwr as fl
from flwr.common import FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from pyarrow import feather

def create_baseline_strategy(
    parent_strategy: fl.server.strategy,
    initial_parameters: Parameters,
):

    class BaselineContribution(parent_strategy):
        def __init__(
            self,
            initial_parameters: Parameters = initial_parameters,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            sampler: flcon.Sampler = flcon.MonteCarloSampler,
            sample_ratio: float = 0.5,
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
            super().__init__()

            self.evaluate_fn = evaluate_fn
            self.results: List[Tuple[ClientProxy, FitRes]] = None
            self.on_fit_config_fn = on_fit_config_fn

            self.initial_parameters = initial_parameters
            self.fraction_fit = fraction_fit
            self.fraction_evaluate = fraction_evaluate
            self.min_fit_clients = min_fit_clients
            self.min_evaluate_clients = min_evaluate_clients
            self.min_available_clients = min_available_clients

            self.sampler = sampler
            self.sample_ratio = sample_ratio
            self.df = []

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using weighted average."""
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Calculate contributions of clients using flwr_contributions
            flcon.multi_round_reconstruction(server_round,
                                             results,
                                             failures,
                                             self.evaluate_fn,
                                             super().aggregate_fit,
                                             self.sampler,
                                             self.sample_ratio
                                             )

            return super().aggregate_fit(
                server_round, results, failures
            )

    return BaselineContribution