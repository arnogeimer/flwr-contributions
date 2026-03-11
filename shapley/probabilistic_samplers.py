"""Probabilistic / Monte Carlo permutation samplers.

Given a client set [0, ..., d-1] and a sample size, generate permutations
of the client set.
Output: list of tuples, each a permutation of clients.

Samplers are implementations of https://arxiv.org/abs/2104.12199
Using Samplers in FL has been proposed in https://arxiv.org/pdf/2109.02053
"""

import warnings
from itertools import permutations
from math import ceil, factorial
from typing import List

import numpy as np

Permutation = tuple[int, ...]


def inverse(pi: Permutation) -> Permutation:
    """Return the inverse permutation such that pi * inverse(pi) = identity.

    Treats pi as a bijection on its element set: sorted[i] -> pi[i].
    The inverse maps pi[i] -> sorted[i], applied to the sorted elements.
    """
    sorted_elements = sorted(pi)
    sigma_inv = {pi[i]: sorted_elements[i] for i in range(len(pi))}
    return tuple(sigma_inv[e] for e in sorted_elements)


class Sampler(object):
    def __init__(self, **kwargs) -> None:
        self.samples = None
        self.samplesize = kwargs.get("samplesize", 10)
        self.seed = kwargs.get("seed", 1)

    def generate_samples(self, results: List[int]):
        if self.samplesize <= 0:
            self.samples = []
            return
        self._generate_samples(results)

    def add_antithetic_samples(self):
        """Add the inverse of each permutation."""
        expanded = set(self.samples)
        for entry in self.samples:
            expanded.add(inverse(entry))
        self.samples = list(expanded)


class FullSampler(Sampler):
    """Enumerate all permutations of the client set."""

    MAX_CLIENTS = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate_samples(self, results: List[int]):
        n = len(results)
        if n > self.MAX_CLIENTS:
            raise ValueError(
                f"FullSampler with {n} clients would generate "
                f"{factorial(n)} permutations. Use an approximate sampler instead."
            )
        elif n > 8:
            warnings.warn(
                f"FullSampler with {n} clients generates "
                f"{factorial(n)} permutations — this may be slow."
            )
        self.samples = list(permutations(results))


class MonteCarloSampler(Sampler):
    """Uniform random permutation sampler."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate_samples(self, results: List[int]):
        rng = np.random.default_rng(self.seed)
        samples = []
        for _ in range(self.samplesize):
            arr = list(results)
            rng.shuffle(arr)
            samples.append(tuple(arr))
        self.samples = samples


class AntitheticMonteCarloSampler(MonteCarloSampler):
    """Antithetic Monte Carlo: generate n/2 permutations + their inverses.

    For odd samplesize, generates ceil(n/2) base + their inverses (n+1 total).
    """

    def __init__(self, **kwargs):
        original_size = kwargs.get("samplesize", 10)
        kwargs["samplesize"] = ceil(original_size / 2)
        super().__init__(**kwargs)

    def _generate_samples(self, results: List[int]):
        super()._generate_samples(results)
        self.add_antithetic_samples()


class StratifiedSampler(Sampler):
    """Stratified: split clients into subsets, generate random permutations from each."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate_samples(self, results: List[int]):
        rng = np.random.default_rng(self.seed)
        n = len(results)

        # Determine number of splits so each subset is small enough
        n_splits = max(1, n // 4)
        shuffled = list(results)
        rng.shuffle(shuffled)

        # Split into roughly equal parts
        splits = [shuffled[i * n // n_splits : (i + 1) * n // n_splits]
                  for i in range(n_splits)]

        # Generate random permutations from each split
        per_split = max(1, self.samplesize // n_splits)
        samples = []
        for split in splits:
            for _ in range(per_split):
                arr = list(split)
                rng.shuffle(arr)
                samples.append(tuple(arr))
        self.samples = samples
