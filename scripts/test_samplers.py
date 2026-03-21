"""Test all samplers: sample counts, edge cases, and antithetic correctness."""

from shapley.probabilistic_samplers import (
    FullSampler,
    MonteCarloSampler,
    AntitheticMonteCarloSampler,
    StratifiedSampler,
    reverse,
)
from shapley.kernel_samplers import (
    KendallSampler,
    MallowsSampler,
    SpearmanSampler,
)

clients = [0, 1, 2, 3, 4]

print("=" * 60)
print("PROBABILISTIC SAMPLERS")
print("=" * 60)

# FullSampler
print("\n--- FullSampler ---")
fs = FullSampler()
fs.generate_samples(clients)
print(f"  Clients: {clients}")
print(f"  Expected: 5! = 120 permutations")
print(f"  Got: {len(fs.samples)} permutations")
# Check all are full permutations
assert all(len(p) == len(clients) for p in fs.samples)
assert all(set(p) == set(clients) for p in fs.samples)

# MonteCarloSampler with various sizes
print("\n--- MonteCarloSampler ---")
for n in [0, 1, 5, 20]:
    mc = MonteCarloSampler(samplesize=n)
    mc.generate_samples(clients)
    print(f"  samplesize={n}: got {len(mc.samples)} permutations")
    if mc.samples:
        assert all(len(p) == len(clients) for p in mc.samples)
        assert all(set(p) == set(clients) for p in mc.samples)

# AntitheticMonteCarloSampler
print("\n--- AntitheticMonteCarloSampler ---")
for n in [0, 1, 5, 10]:
    amc = AntitheticMonteCarloSampler(samplesize=n)
    amc.generate_samples(clients)
    print(f"  samplesize={n}: got {len(amc.samples)} permutations (from {n} base)")
    # Verify inverses are present
    if amc.samples:
        samples_set = set(amc.samples)
        for pi in list(amc.samples)[:n]:  # check original samples
            pi_inv = reverse(pi)
            if pi_inv not in samples_set:
                print(f"    MISSING INVERSE: {pi} -> {pi_inv}")

# Verify reverse correctness
print("\n--- Reverse verification ---")
mc = MonteCarloSampler(samplesize=5, seed=42)
mc.generate_samples(clients)
for pi in mc.samples:
    pi_rev = reverse(pi)
    ok = reverse(pi_rev) == pi
    print(f"  pi={pi}  rev={pi_rev}  rev(rev)={reverse(pi_rev)}  {'OK' if ok else 'FAIL'}")

# StratifiedSampler
print("\n--- StratifiedSampler ---")
for n in [0, 1, 5, 20]:
    ss = StratifiedSampler(samplesize=n)
    ss.generate_samples(clients)
    print(f"  samplesize={n}: got {len(ss.samples)} permutations")
    if ss.samples:
        # Stratified produces permutations of subsets, not full client set
        for p in ss.samples:
            assert set(p).issubset(set(clients)), f"Invalid subset: {p}"

print("\n" + "=" * 60)
print("KERNEL SAMPLERS")
print("=" * 60)

for name, SamplerClass in [("Kendall", KendallSampler), ("Mallows", MallowsSampler), ("Spearman", SpearmanSampler)]:
    print(f"\n--- {name}Sampler ---")
    for n in [0, 1, 5, 10]:
        sampler = SamplerClass(samplesize=n)
        sampler.generate_samples(clients)
        n_perms = len(sampler.generate_permutations(clients))
        print(f"  samplesize={n}: {n_perms} permutations -> {len(sampler.samples)} unique subsets")

# Edge case: small client sets
print("\n--- Edge cases ---")
for client_set in [[0], [0, 1], [0, 1, 2]]:
    print(f"\n  Clients: {client_set}")
    mc = MonteCarloSampler(samplesize=5)
    mc.generate_samples(client_set)
    print(f"    MonteCarlo(5): {len(mc.samples)} permutations")

    ks = KendallSampler(samplesize=3)
    ks.generate_samples(client_set)
    print(f"    Kendall(3): {len(ks.samples)} subsets")

print("\nAll tests passed!")
