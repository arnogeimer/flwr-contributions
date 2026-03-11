"""Permutation-based Shapley value computation.

Given permutation samples and an evaluation function, compute Shapley values
for each client using the permutation-based formula:

    φ_i = (1/m) Σ_{π} [v(P_i^π ∪ {i}) - v(P_i^π)]

where P_i^π is the set of clients preceding client i in permutation π,
and v is the evaluation (utility) function.
"""

from typing import Callable, Dict, List

from shapley.probabilistic_samplers import Sampler

Permutation = tuple[int, ...]


def shapley_values(
    client_ids: List[int],
    permutations: List[Permutation],
    evaluate_fn: Callable[[tuple[int, ...]], float],
) -> Dict[int, float]:
    """Compute Shapley values from permutation samples.

    Walks along each permutation in order, reusing the previous v_with as the
    next v_without. Caches evaluations to avoid redundant calls across permutations.

    Parameters
    ----------
    client_ids : List[int]
        The full set of client IDs.
    permutations : List[Permutation]
        Sampled permutations of the client set.
    evaluate_fn : Callable[[tuple[int, ...]], float]
        Utility function: given a sorted tuple of client IDs, returns a score.

    Returns
    -------
    Dict[int, float]
        Mapping from client ID to its estimated Shapley value.
    """
    svs: Dict[int, float] = {i: 0.0 for i in client_ids}
    cache: Dict[tuple[int, ...], float] = {}
    m = len(permutations)

    def v(subset: tuple[int, ...]) -> float:
        if subset not in cache:
            cache[subset] = evaluate_fn(subset)
        return cache[subset]

    for pi in permutations:
        current = []
        for pos, j in enumerate(pi):
            current.append(j)
            subset = tuple(sorted(current))
            v_curr = v(subset)
            if pos > 0:
                # Skip first position: no fair baseline to compare against
                svs[j] += v_curr - v_prev
            v_prev = v_curr

    # Divide by number of permutations where each client was NOT first
    first_counts: Dict[int, int] = {i: 0 for i in client_ids}
    for pi in permutations:
        first_counts[pi[0]] += 1
    for j in client_ids:
        denom = m - first_counts[j]
        if denom > 0:
            svs[j] /= denom

    return svs


def shapley_values_from_sampler(
    client_ids: List[int],
    sampler: Sampler,
    evaluate_fn: Callable[[tuple[int, ...]], float],
) -> Dict[int, float]:
    """Convenience: generate permutations from a sampler, then compute Shapley values.

    Parameters
    ----------
    client_ids : List[int]
        The full set of client IDs.
    sampler : Sampler
        A sampler instance (already configured with samplesize, seed, etc.).
    evaluate_fn : Callable[[tuple[int, ...]], float]
        Utility function: given a sorted tuple of client IDs, returns a score.

    Returns
    -------
    Dict[int, float]
        Mapping from client ID to its estimated Shapley value.
    """
    sampler.generate_samples(client_ids)
    return shapley_values(client_ids, sampler.samples, evaluate_fn)
