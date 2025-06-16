import random
from itertools import chain, combinations
from typing import Callable, List, Tuple, Union

import numpy as np
from flwr.common import FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

""" Samplers are implementations of https://arxiv.org/abs/2104.12199 """

class Sampler(object):
    def __init__(self, results: List[int]) -> None:
        self.results = results
        self.pws = self.create_powerset(self.results)

    def create_powerset(self, set: List[int]) -> chain:
        "Returns the powerset of a set"
        pws = chain.from_iterable(combinations(set, r) for r in range(1, len(set) + 1))
        return list(pws)

    def add_antithetic_samples(self):
        # adds anti-samples to self.samples
        antisamples = [
            tuple([x for x in self.results if x not in entry]) for entry in self.samples
        ]
        self.samples += antisamples

    def generate_test_samples(self):
        # Returns [entry + i for entry in samples if i not in entry]
        evaluation_samples = []
        self.samples = [entry for entry in self.samples if entry != ()]
        for entry in self.samples:
            for k in self.results:
                if k not in entry:
                    evaluation_samples.append(tuple(sorted(list(entry + (k,)))))
        self.samples = list(set(self.samples + evaluation_samples))


class FullSampler(Sampler):
    # Full powerset sampler

    def __init__(self, results: List[int], **kwargs):
        super().__init__(results)

    def generate_samples(self):
        self.samples = self.pws


class LeaveOneOutSampler(Sampler):
    # Leave-one-out
    
    def __init__(self, results: List[int], **kwargs):
        super().__init__(results)

    def generate_samples(self):
        self.samples = [[j for j in self.results if j != i] for i in self.results] + [self.results]


class MonteCarloSampler(Sampler):
    # Uniform sampler from powerset

    def __init__(self, results: List[int], **kwargs):
        super().__init__(results)
        sample_ratio = kwargs.get("sample_ratio", 0.1)
        self.samplesize = max(int(len(self.pws) * sample_ratio), 1)
        self.seed = kwargs.get("seed", 1)

    def generate_monte_carlo_samples(self):
        random.seed(self.seed)
        self.samples = random.sample(self.pws, k=self.samplesize)

    def generate_samples(self):
        self.generate_monte_carlo_samples()
        self.generate_test_samples()


class AntitheticMonteCarloSampler(MonteCarloSampler):
    # Antithetic: If x is in E, so is !x
    # Antithetic for a Monte Carlo Sampler

    def __init__(self, results: List[int], **kwargs):
        super().__init__(results, **kwargs)

    def generate_samples(self):
        self.generate_monte_carlo_samples()
        self.add_antithetic_samples()
        self.generate_test_samples()


class MultilinearExtensionSampler(Sampler):
    # Samples contain client i with probability pk for 0 < pk < 1

    def __init__(self, results: List[int], **kwargs):
        super().__init__(results)
        sample_ratio = kwargs.get("sample_ratio", 0.05)
        self.samplesize = max(int(len(self.pws) * sample_ratio), 1)
        self.seed = kwargs.get("seed", 1)

    def generate_multilinear_extension_samples(self):
        probabilities = np.linspace(0, 1, self.samplesize + 2)[1:-1]
        samples = []
        random.seed(self.seed)
        for probability in probabilities:
            sampler = [
                (random.random() < probability) for _ in range(len(self.results))
            ]
            if True in sampler:
                samples.append(
                    tuple(
                        [
                            self.results[i]
                            for i in range(len(self.results))
                            if sampler[i]
                        ]
                    )
                )
        self.samples = samples

    def generate_samples(self):
        self.generate_multilinear_extension_samples()
        self.generate_test_samples()


class AntitheticMultilinearExtensionSampler(MultilinearExtensionSampler):
    # Antithetic: If x is in E, so is !x
    # Antithetic for a Multilinear Extension Sampler

    def __init__(self, results: List[int], **kwargs):
        super().__init__(results, **kwargs)

    def generate_samples(self):
        self.generate_multilinear_extension_samples()
        self.add_antithetic_samples()
        self.generate_test_samples()


class StratifiedSampler(Sampler):
    # E = pws(D_1) u pws(D_2) u pws(D3) u ... , where D = D1 |u| D2 |u| D3 ...

    def __init__(self, results: List[int], **kwargs):
        super().__init__(results)
        sample_ratio = kwargs.get("sample_ratio", 0.05)
        self.samplesize = max(int(len(self.pws) * sample_ratio), 1)
        self.seed = kwargs.get("seed", 1)

        np.random.seed(self.seed)
        np.random.shuffle(self.results)

    def generate_samples(self):
        splits = [
            self.results[i : i + self.samplesize]
            for i in range(0, len(self.results), self.samplesize)
        ]
        samples = [self.create_powerset(entry) for entry in splits]
        self.samples = [tuple(set(x)) for subset in samples for x in subset]
        self.generate_test_samples()


def multi_round_reconstruction(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    evaluate_fn: Callable,
    aggregate_fit: Callable = FedAvg.aggregate_fit,
    sampler: Sampler = FullSampler,
    sample_ratio: float = 0.5,
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
    sampler = sampler(result_ids, sample_ratio=sample_ratio, seed=server_round)
    sampler.generate_samples()
    if len(sampler.samples) == 0:
        return None
    shapley_dict: List = []
    for subset in sampler.samples:
        sub_results = [results_sorted[i] for i in subset]
        parameters_aggregated, _ = aggregate_fit(server_round, sub_results, failures)
        weights_aggregated = parameters_to_ndarrays(parameters_aggregated)
        loss, metrics = evaluate_fn(server_round, weights_aggregated, {})
        shapley_dict.append([subset, loss, metrics["accuracy"]])
    return shapley_dict

def one_round_reconstruction(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    evaluate_fn: Callable,
    aggregate_fit: Callable = FedAvg.aggregate_fit,
    samples: Sampler.samples = [],
    initial_parameters = None,
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
        for (subset, gradient) in zip(samples, gradients):
            final_model = parameters_to_ndarrays(init_params - gradient)
            loss, metrics = evaluate_fn(server_round, final_model, {})
            shapley_dict.append([subset, loss, metrics["accuracy"]])
        return None, None, shapley_dict

    else:
        return initial_parameters, gradients, []
