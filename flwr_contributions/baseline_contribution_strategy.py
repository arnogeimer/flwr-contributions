from flwr_contributions import shapley_values as flcon
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)
import flwr as fl
from flwr.common import FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from pyarrow import feather

def create_baseline_strategy(
    parent_strategy: fl.server.strategy,
    sampler: flcon.Sampler = flcon.MonteCarloSampler,
):

    '''
    Example usage:
    
    from flwr_contributions.baseline_strategy import create_baseline_strategy
    import flwr_contributions.shapley_values as flcon

    sampler = flcon.MonteCarloSampler
    sampler.samplesize = 1
    sampler.seed = 1

    def start_simulation():
        strategy = create_baseline_strategy(FedAvg, sampler)
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=10,
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy(
                initial_parameters=get_initial_parameters(),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=fit_config,
            ),
        )
    '''

    class Baseline(parent_strategy):
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
                                             )
            return super().aggregate_fit(
                server_round, results, failures
            )

    return Baseline