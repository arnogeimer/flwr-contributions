from typing import Callable, List, Tuple, Union

from flwr.common import FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

def leave_one_out_contributions(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    evaluate_fn: Callable,
    aggregate_fit: Callable = FedAvg.aggregate_fit,
) -> Tuple[int, Tuple[int], float, float]:

    """
    A function calculating Leave-one-out contributions (source: https://arxiv.org/pdf/1909.08525).
    Is plugged into aggregate_fit in the flwr.strategy.
    """

    if not results:
        return None
    if evaluate_fn is None:
        return None

    # Sort results by id
    result_ids = [fitres.metrics["client_id"] for (_, fitres) in results]
    results_sorted = [result for _, result in sorted(zip(result_ids, results))]
    # Get a sample of clients for evaluation. Seeded by server round for reproducibility.
    accuracies: List = []
    # Get the full loss and accuracy
    parameters_aggregated, _ = aggregate_fit(server_round, results_sorted, failures)
    weights_aggregated = parameters_to_ndarrays(parameters_aggregated)
    loss, metrics = evaluate_fn(server_round, weights_aggregated, {})
    accuracies.append([result_ids, loss, metrics["accuracy"]])
    for id in result_ids:
        sub_results = [results_sorted[i] for i in result_ids if i is not id]
        parameters_aggregated, _ = aggregate_fit(server_round, sub_results, failures)
        weights_aggregated = parameters_to_ndarrays(parameters_aggregated)
        loss, metrics = evaluate_fn(server_round, weights_aggregated, {})
        accuracies.append([sub_results, loss, metrics["accuracy"]])
    return accuracies
