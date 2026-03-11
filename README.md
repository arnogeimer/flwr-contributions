# flwr-contributions

Shapley-value-based client contribution measurement for the [Flower](https://flower.ai) federated learning framework.

This work was used in the paper [On the Volatility of Shapley-Based Contribution Metrics in Federated Learning](https://arxiv.org/abs/2405.08044), presented at IJCNN 2025.

---

## What this is

In federated learning, clients contribute heterogeneously — different data distributions, quantities, and qualities. This library measures each client's contribution to the global model using Shapley values, which fairly attribute model performance across participants.

Shapley values are computed via the permutation-based formula:

$$\phi_i = \frac{1}{m} \sum_{\pi} \left[ v(P_i^\pi \cup \{i\}) - v(P_i^\pi) \right]$$

where $v$ is a valuation function (e.g. accuracy on a held-out test set), $\pi$ ranges over sampled permutations, and $P_i^\pi$ is the set of clients preceding $i$ in $\pi$.

---

## Samplers

Exact Shapley computation requires enumerating all $n!$ permutations, which is infeasible for large $n$. We support approximate samplers from [Kolpaczki et al. (2024)](https://arxiv.org/pdf/2104.12199):

| Sampler | Description |
|---|---|
| `FullSampler` | Enumerates all permutations (exact, $n \leq 10$) |
| `MonteCarloSampler` | Uniform random permutations |
| `AntitheticMonteCarloSampler` | Monte Carlo + inverse permutations for variance reduction |
| `StratifiedSampler` | Samples from stratified client subgroups |
| `KendallSampler` | Greedy kernel-based selection using Kendall's tau distance |
| `MallowsSampler` | Greedy kernel-based selection using the Mallows kernel |
| `SpearmanSampler` | Greedy kernel-based selection using Spearman's rho distance |

Kernel-based samplers greedily maximise diversity among selected permutations, improving estimation quality for a fixed sample budget.

---

## FL Contribution Strategies

Two reconstruction strategies from [Song et al. (2019)](https://ieeexplore.ieee.org/document/9006327):

**One-round** (`OneRoundReconstructor`): Permutations are sampled once. Per-client gradient updates are accumulated across all rounds. At the end, subset models are reconstructed by combining accumulated gradients and evaluated once.

**Multi-round** (`MultiRoundReconstructor`): Each round, fresh permutations are sampled and subsets are aggregated and evaluated independently. Per-round Shapley values are collected; the full history is returned at the end for flexible aggregation.

Both are plugged into Flower via a `ContributionStrategy` wrapping `FedAvg`.

---

## Repo structure

```
flwr-contributions/
├── shapley/                    # Sampler and Shapley value logic
│   ├── probabilistic_samplers.py   # Monte Carlo, antithetic, stratified
│   ├── kernel_samplers.py          # Kendall, Mallows, Spearman kernel samplers
│   ├── sphere_samplers.py          # Sphere-based samplers
│   └── shapley_values.py           # Permutation-based Shapley formula with caching
├── flwr_shapley/               # Flower integration
│   ├── federated_shapley.py        # OneRoundReconstructor, MultiRoundReconstructor
│   └── baseline_contribution_strategy.py
├── scripts/                    # Entry points
│   ├── server_app.py               # ServerApp with ContributionStrategy
│   └── client_app.py               # ClientApp (training + data loading)
├── task/                       # Dataset/model definitions
│   ├── cifar10.py
│   ├── cuda_vision.py
│   └── data_utils.py
├── tests/                      # Unit tests
└── pyproject.toml              # Dependencies and federation config
```

---

## Running

```bash
uv run flwr run .
```

Federation and sampler settings are configured in `pyproject.toml` under `[tool.flwr.app.config]`.
