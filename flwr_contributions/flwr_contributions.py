from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr.server.strategy
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy



def flwr_contribution_strategy(strategy: flwr.server.strategy, contribution_method: Callable):
    class flwr_contribution(strategy):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}
            contributions = contribution_method(
                server_round=server_round,
                results=results,
                failures=failures,
                evaluate_fn=self.evaluate_fn,
                aggregate_fit=super().aggregate_fit,
            )
            print(contributions)
            return super().aggregate_fit(server_round, results, failures)

    return flwr_contribution
