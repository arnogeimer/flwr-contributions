import warnings
from itertools import chain, combinations
from typing import Callable, List, Tuple, Union

import numpy as np
from flwr.common import FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

""" Samplers are implementations of https://arxiv.org/abs/2104.12199 """
""" Using Samplers in FL has been proposed in https://arxiv.org/pdf/2109.02053 """


def int_to_set(x: int, results: List[int]) -> tuple[int]:
    bin = format(x, "b").rjust(len(results), "0")
    sample = tuple([results[i] for i, entry in enumerate(bin) if entry == "1"])
    return sample


class Sampler(object):
    def __init__(self, ) -> None:
        self.samples = None

    def generate_test_samples(self, results: List[int]):
        # Returns [entry + i for entry in samples if i not in entry]
        evaluation_samples = []
        self.samples = [entry for entry in self.samples if entry != ()]
        for entry in self.samples:
            for k in results:
                if k not in entry:
                    evaluation_samples.append(tuple(sorted(list(entry + (k,)))))
        self.samples = list(set(self.samples + evaluation_samples))

    def add_antithetic_samples(self, results: List[int]):
        # adds anti-samples to self.samples
        antisamples = [
            tuple([x for x in results if x not in entry]) for entry in self.samples
        ]
        self.samples += antisamples


class FullSampler(Sampler):
    # Full powerset sampler

    def __init__(self, **kwargs):
        super().__init__()

    def generate_samples(self, results: List[int]):
        warnings.warn(
            """Warning: It is not recommended to use the full sampler for high amount of clients.
                         Due to exponential growth of powerset sizes, your system may kill the process."""
        )
        self.samples = list(
            chain.from_iterable(
                combinations(results, r) for r in range(1, len(results) + 1)
            )
        )


class LeaveOneOutSampler(Sampler):
    # Leave-one-out

    def __init__(self, **kwargs):
        super().__init__()

    def generate_samples(self, results: List[int]):
        self.samples = sorted(
            [[j for j in results if j != i] for i in results] + [results]
        )


class MonteCarloSampler(Sampler):
    # Uniform sampler from powerset

    def __init__(self, **kwargs):
        super().__init__()
        self.samplesize = kwargs.get("samplesize", 10)
        self.seed = kwargs.get("seed", 1)

    def generate_monte_carlo_samples(self, results: List[int]):
        # Due to the computational complexity of power sets, we uniformly sample powerset indices and fit the subset through the binary representation
        np.random.seed(self.seed)
        self.samples = [
            int_to_set(entry, results)
            for entry in np.random.randint(0, 2 ** len(results), self.samplesize)
        ]

    def generate_samples(self, results: List[int]):
        self.generate_monte_carlo_samples(results)
        self.generate_test_samples(results)


class AntitheticMonteCarloSampler(MonteCarloSampler):
    # Antithetic: If x is in E, so is !x
    # Antithetic for a Monte Carlo Sampler

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_samples(self, results: List[int]):
        self.generate_monte_carlo_samples(results)
        self.add_antithetic_samples(results)
        self.generate_test_samples(results)


class MultilinearExtensionSampler(Sampler):
    # Samples contain client i with probability pk for 0 < pk < 1

    def __init__(self, **kwargs):
        super().__init__()
        self.samplesize = kwargs.get("samplesize", 10)
        self.seed = kwargs.get("seed", 1)

    def generate_multilinear_extension_samples(self, results: List[int]):
        probabilities = np.linspace(0, 1, self.samplesize + 2)[1:-1]
        samples = []
        np.random.seed(self.seed)
        for probability in probabilities:
            sampler = [
                (np.random.rand() < probability) for _ in range(len(results))
            ]
            if True in sampler:
                samples.append(
                    tuple(
                        [
                            results[i]
                            for i in range(len(results))
                            if sampler[i]
                        ]
                    )
                )
        self.samples = samples

    def generate_samples(self, results: List[int], ):
        self.generate_multilinear_extension_samples(results)
        self.generate_test_samples(results)


class AntitheticMultilinearExtensionSampler(MultilinearExtensionSampler):
    # Antithetic: If x is in E, so is !x
    # Antithetic for a Multilinear Extension Sampler

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_samples(self, results: List[int], ):
        self.generate_multilinear_extension_samples(results)
        self.add_antithetic_samples(results)
        self.generate_test_samples(results)


class StratifiedSampler(Sampler):
    # E = pws(D_1) u pws(D_2) u pws(D3) u ... , where D = D1 |u| D2 |u| D3 ...

    def __init__(self, **kwargs):
        super().__init__()
        self.samplesize = kwargs.get("samplesize", 10)
        self.seed = kwargs.get("seed", 1)
        # We search for the optimal split to account for the samplesize the user specifies.
        # By splitting into smaller and smaller subsets, we decrease the total amount of samples.
        # With find_best_split_size, we search for the amount of subsets which is just below the desired sample size.
        # Note that, for too small sample sizes (below len(results)), we have to ignore "sample_number", as a stratified sampling would be impossible.

    def find_best_split_size(self, results: List[int]):
        k = len(results)
        sizes = np.array(
            [
                (i - (k % i)) * (2 ** (int(k / i)))
                + (k % i) * (2 ** (int(k / i) + 1))
                for i in range(1, len(results))
            ]
        )
        try:
            best_i = min(np.flatnonzero(sizes < self.samplesize)) + 1
        except Exception:
            best_i = k
        return np.cumsum(
            [0]
            + (best_i - (k % best_i)) * [int(k / best_i)]
            + (k % best_i) * [int(k / best_i) + 1]
        )

    def generate_stratified_samples(self, results: List[int]):
        self.split_sets = self.find_best_split_size(results)

        np.random.seed(self.seed)
        np.random.shuffle(self.results)
        splits = [
            self.results[self.split_sets[i] : self.split_sets[i + 1]]
            for i in range(0, len(self.split_sets) - 1)
        ]
        samples = [
            list(
                chain.from_iterable(
                    combinations(entry, r) for r in range(1, len(entry) + 1)
                )
            )
            for entry in splits
        ]
        self.samples = [tuple(set(x)) for subset in samples for x in subset]

    def generate_samples(self, results: List[int]):
        self.generate_stratified_samples(results)
        self.generate_test_samples(results)


def multi_round_reconstruction(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    evaluate_fn: Callable,
    aggregate_fit: Callable = FedAvg.aggregate_fit,
    sampler: Sampler = FullSampler,
) -> Tuple[int, Tuple[int], float, float]:
    """
    A function calculating multi-round Shapley-value contributions, based on results sent to the server (source: https://arxiv.org/pdf/1909.08525).
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
    sampler = sampler(seed=server_round)
    sampler.generate_samples(result_ids)
    if len(sampler.samples) == 0:
        return None
    shapley_dict: List = []
    for subset in sampler.samples:
        sub_results = [results_sorted[i] for i in subset]
        parameters_aggregated, _ = aggregate_fit(server_round, sub_results, failures)
        weights_aggregated = parameters_to_ndarrays(parameters_aggregated)
        loss, metrics = evaluate_fn(server_round, weights_aggregated, {})
        shapley_dict.append([subset, loss, metrics["accuracy"]])
        print(subset, loss, metrics["accuracy"])
    return shapley_dict


def one_round_reconstruction(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    evaluate_fn: Callable,
    aggregate_fit: Callable = FedAvg.aggregate_fit,
    samples: List[List[int]] = [],
    initial_parameters=None,
    final_round: int = 0,
) -> Tuple[int, Tuple[int], float, float]:
    """
    A function calculating one-round Shapley-value contributions, based on results sent to the server (source: https://arxiv.org/pdf/1909.08525).
    Is plugged into aggregate_fit in the flwr.strategy.
    """

    gradients = None
    init_params = initial_parameters

    if not results:
        return None
    if evaluate_fn is None:
        return None
    if len(samples) == 0:
        return None

    # Sort results by id
    result_ids = [fitres.metrics["client_id"] for (_, fitres) in results]
    results_sorted = [result for _, result in sorted(zip(result_ids, results))]
    for i, subset in enumerate(samples):
        sub_results = [results_sorted[i] for i in subset]
        parameters_aggregated, _ = aggregate_fit(server_round, sub_results, failures)
        if gradients is None:
            gradients.append(initial_parameters - parameters_aggregated)
        else:
            gradients[i] += initial_parameters - parameters_aggregated

    initial_parameters = aggregate_fit(server_round, results, failures)

    if server_round == final_round:
        shapley_dict: List = []
        for subset, gradient in zip(samples, gradients):
            final_model = parameters_to_ndarrays(init_params - gradient)
            loss, metrics = evaluate_fn(server_round, final_model, {})
            shapley_dict.append([subset, loss, metrics["accuracy"]])
        return None, None, shapley_dict

    else:
        return initial_parameters, gradients, []
