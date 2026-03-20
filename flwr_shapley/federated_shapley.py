"""Federation-specific Shapley value reconstruction.

These functions plug into Flower's aggregate_fit to compute Shapley values
using federated learning results (ClientProxy, FitRes, etc.).

Two approaches:
- OneRoundReconstructor: sample permutations once, accumulate per-client gradients
  across rounds, evaluate subsets at the end using accumulated gradients.
- multi_round_reconstruction: sample and evaluate subsets independently each round.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from shapley.probabilistic_samplers import FullSampler, Sampler
from shapley.shapley_values import shapley_values

logger = logging.getLogger(__name__)


class OneRoundReconstructor:
    """Accumulate per-client gradients across rounds, compute Shapley values at the end.

    Usage: create once, call `one_round` from aggregate_fit each round.
    On the final round, `one_round` returns a Dict[int, float] of Shapley values.

    The approach:
    1. Sample permutations once (first round).
    2. Each round: extract per-client gradient (client_params - global_params),
       accumulate per client.
    3. Final round: for each subset, reconstruct a model by combining accumulated
       gradients (weighted by num_examples), evaluate it.
       Feed into the permutation-based Shapley formula.
    """

    def __init__(
        self,
        sampler: Sampler,
        evaluate_fn: Callable,
        num_rounds: int,
    ):
        self.sampler = sampler
        self.evaluate_fn = evaluate_fn
        self.num_rounds = num_rounds
        self.permutations: List[tuple[int, ...]] = []
        self.client_ids: List[int] = []
        # Per-client accumulated gradients and total example counts
        self.client_gradients: Dict[int, NDArrays] = {}
        self.client_num_examples: Dict[int, int] = {}
        self.initial_parameters: NDArrays = None
        # client_id -> list of Shapley values (single entry for one-round)
        self.history: Dict[int, List[float]] = {}

    def one_round(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        current_parameters: NDArrays,
    ) -> Optional[Dict[int, List[float]]]:
        """Process one round of results. Returns Shapley value history on the final round."""
        if not results:
            return None

        # First round: save initial parameters and generate permutations
        if server_round == 1:
            self.initial_parameters = current_parameters
            self.client_ids = sorted(
                fitres.metrics["client_id"] for _, fitres in results
            )
            self.sampler.generate_samples(self.client_ids)
            self.permutations = self.sampler.samples

        # Accumulate per-client gradients
        for _, fitres in results:
            client_id = fitres.metrics["client_id"]
            client_params = parameters_to_ndarrays(fitres.parameters)
            gradient = [
                c - g for c, g in zip(client_params, current_parameters)
            ]

            if client_id not in self.client_gradients:
                self.client_gradients[client_id] = gradient
                self.client_num_examples[client_id] = fitres.num_examples
            else:
                self.client_gradients[client_id] = [
                    acc + g
                    for acc, g in zip(self.client_gradients[client_id], gradient)
                ]
                self.client_num_examples[client_id] += fitres.num_examples

        # Final round: compute Shapley values and store in history
        if server_round == self.num_rounds:
            final_values = self._compute_shapley()
            for client_id, value in final_values.items():
                self.history.setdefault(client_id, []).append(value)
            return self.history

        return None

    def _compute_shapley(self) -> Dict[int, float]:
        """Reconstruct models for each subset and compute Shapley values."""

        def evaluate_subset(subset: tuple[int, ...]) -> float:
            # Weighted average of accumulated gradients for clients in subset
            total_examples = sum(
                self.client_num_examples[cid] for cid in subset
            )
            combined: NDArrays = None
            for cid in subset:
                weight = self.client_num_examples[cid] / total_examples
                grad = self.client_gradients[cid]
                if combined is None:
                    combined = [weight * g for g in grad]
                else:
                    combined = [c + weight * g for c, g in zip(combined, grad)]

            # Reconstruct model: initial + combined gradient
            model = [p + g for p, g in zip(self.initial_parameters, combined)]
            loss, metrics = self.evaluate_fn(self.num_rounds, model, {})
            accuracy = metrics["accuracy"]
            print(f"[shapley] subset {subset} -> accuracy: {round(100 * accuracy, 2)}%")
            return accuracy

        return shapley_values(self.client_ids, self.permutations, evaluate_subset)


class MultiRoundReconstructor:
    """Compute Shapley values each round by evaluating subset aggregations independently.

    Each round: sample fresh permutations, aggregate each subset, evaluate,
    and compute Shapley values via the permutation-based formula.
    Collects per-round values; returns full history on the final round.
    """

    def __init__(
        self,
        sampler: Sampler,
        evaluate_fn: Callable,
        num_rounds: int,
        aggregate_fit: Callable = FedAvg.aggregate_fit,
    ):
        self.sampler = sampler
        self.evaluate_fn = evaluate_fn
        self.num_rounds = num_rounds
        self.aggregate_fit = aggregate_fit
        # client_id -> list of per-round Shapley values
        self.history: Dict[int, List[float]] = {}

    def on_round(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[Dict[int, List[float]]]:
        """Compute Shapley values for this round, accumulate history.

        Returns the full per-round history Dict[int, List[float]] on the
        final round, None otherwise.
        """
        if not results:
            return None

        result_ids = [fitres.metrics["client_id"] for (_, fitres) in results]
        results_by_id = {
            fitres.metrics["client_id"]: (proxy, fitres)
            for proxy, fitres in results
        }

        self.sampler.generate_samples(result_ids)
        if not self.sampler.samples:
            return None

        def evaluate_subset(subset: tuple[int, ...]) -> float:
            sub_results = [results_by_id[cid] for cid in subset]
            parameters_aggregated, _ = self.aggregate_fit(server_round, sub_results, failures)
            weights_aggregated = parameters_to_ndarrays(parameters_aggregated)
            loss, metrics = self.evaluate_fn(server_round, weights_aggregated, {})
            accuracy = metrics["accuracy"]
            print(f"[shapley] subset {subset} -> accuracy: {round(100 * accuracy, 2)}%")
            return accuracy

        round_values = shapley_values(result_ids, self.sampler.samples, evaluate_subset)
        for client_id, value in round_values.items():
            self.history.setdefault(client_id, []).append(value)

        if server_round == self.num_rounds:
            return self.history

        return None
