"""Kernel-based permutation samplers.

Given a client set and a sample size, generate permutations of the client set
using kernel-based greedy selection for maximum separation.
Output: list of tuples, each a full permutation of the clients.
"""

from math import comb, exp
from typing import Callable, List

import numpy as np

Permutation = tuple[int, ...]


def n_con(x: Permutation, y: Permutation) -> int:
    """Count concordant pairs between two permutations of the same elements.

    A pair (i, j) with i < j is concordant if x and y agree on the relative
    ordering: (x[i] - x[j]) and (y[i] - y[j]) have the same sign.
    """
    n = len(x)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                count += 1
    return count


def n_dis(x: Permutation, y: Permutation) -> int:
    """Count discordant pairs between two permutations of the same elements.

    A pair (i, j) with i < j is discordant if x and y disagree on the relative
    ordering: (x[i] - x[j]) and (y[i] - y[j]) have opposite signs.
    """
    n = len(x)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) < 0:
                count += 1
    return count


def kendall_kernel(x: Permutation, y: Permutation) -> float:
    """Kendall tau kernel."""
    d = comb(len(x), 2)
    if d == 0:
        return 0.0
    return (n_con(x, y) - n_dis(x, y)) / d


def mallows_kernel(x: Permutation, y: Permutation, theta: float = 1.0) -> float:
    """Mallows kernel parameterized by theta."""
    d = comb(len(x), 2)
    if d == 0:
        return 1.0
    return exp((-theta * n_dis(x, y)) / d)


def spearman_kernel(x: Permutation, y: Permutation) -> float:
    """Spearman rho kernel: inner product x^T y."""
    return sum(xi * yi for xi, yi in zip(x, y))


class KernelSampler:
    """Base class for kernel-based permutation samplers.

    Generates permutations selected for maximum separation under a given
    kernel function. The first permutation is random; each subsequent one
    is chosen (from a pool of candidates) to minimize the total kernel
    similarity to all previously selected permutations.

    Output is a list of full permutations (tuples preserving order).

    Subclasses must set `self.distance_fn` to one of the kernel functions above.
    """

    def __init__(self, **kwargs):
        self.samples = None
        self.samplesize = kwargs.get("samplesize", 5)
        self.seed = kwargs.get("seed", 1)
        self.n_candidates = kwargs.get("n_candidates", 20) # number of random candidates to consider at each step
        self.d: int = 0  # set in generate_permutations from len(results)
        self.distance_fn: Callable[[Permutation, Permutation], float] = None
        self._expected_distance_fn: Callable[[int], float] = None
        self._expected_distance_val: float = 0.0

    def _generate_candidate(self, rng: np.random.Generator, elements: List[int]) -> Permutation:
        """Generate a random permutation of the given elements."""
        arr = list(elements)
        rng.shuffle(arr)
        return tuple(arr)

    def _total_distance_to_selected(self, candidate: Permutation, selected: List[Permutation]) -> float:
        """Sum of distances from candidate to all already-selected permutations."""
        return sum(self.distance_fn(candidate, s) for s in selected)

    def generate_permutations(self, results: List[int]) -> List[Permutation]:
        """Greedy selection: pick permutations maximizing separation from chosen set."""
        self.d = len(results)
        if self.samplesize <= 0:
            return []
        self._expected_distance_val = self._expected_distance_fn(self.d)
        rng = np.random.default_rng(self.seed)

        # Start with one random permutation
        selected: List[Permutation] = [self._generate_candidate(rng, results)]

        for _ in range(1, self.samplesize):
            # Generate a pool of random candidates
            candidates = [
                self._generate_candidate(rng, results)
                for _ in range(self.n_candidates)
            ]
            n = len(selected)
            ed = self._expected_distance_val
            best = max(
                candidates,
                key=lambda c: ed - self._total_distance_to_selected(c, selected) / (n + 1),
            )
            selected.append(best)

        return selected

    def generate_samples(self, results: List[int]):
        """Generate permutations via greedy kernel-based selection."""
        self.samples = self.generate_permutations(results)


class KendallSampler(KernelSampler):
    """Kernel sampler using Kendall tau kernel for maximum separation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance_fn = kendall_kernel
        self._expected_distance_fn = lambda d: 0


class MallowsSampler(KernelSampler):
    """Kernel sampler using Mallows kernel for maximum separation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta = kwargs.get("theta", 1.0)
        self.distance_fn = lambda x, y: mallows_kernel(x, y, self.theta)
        self._expected_distance_fn = self._mallows_expected_distance

    @staticmethod
    def _mallows_expected_distance(d: int, theta: float = 1.0) -> float:
        """E[mallows_kernel(x, y)] for uniform random permutations of d elements.

        Product form: ∏_{i=1}^{d-1} f(i, theta).
        """
        result = 1.0
        for i in range(1, d):
            result *= (1 - exp(-theta * i / comb(d, 2))) / (i * (1 - exp(-theta / comb(d, 2))))
        return result


class SpearmanSampler(KernelSampler):
    """Kernel sampler using Spearman kernel for maximum separation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance_fn = spearman_kernel
        self._expected_distance_fn = lambda d: d * (d + 1) **2 / 4
