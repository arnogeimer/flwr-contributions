#!/usr/bin/env bash
# Run federated learning with each aggregation strategy across multiple seeds.
# MonteCarloSampler is fixed for all runs.

set -euo pipefail

PYPROJECT="pyproject.toml"
STRATEGIES=("fedavg" "fedadam" "fedadagrad" "fedavgm")
SEEDS=("42" "123" "456" "789" "1024")

# Ensure sampler is monte-carlo
sed -i 's/^sampler-type = .*/sampler-type = "monte-carlo"/' "$PYPROJECT"

for seed in "${SEEDS[@]}"; do
    sed -i "s/^seed = .*/seed = $seed/" "$PYPROJECT"

    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "=========================================="
        echo " seed: $seed | strategy: $strategy"
        echo "=========================================="
        echo ""

        sed -i "s/^aggregation-strategy = .*/aggregation-strategy = \"$strategy\"/" "$PYPROJECT"

        flwr run .
    done
done

# Reset to defaults
sed -i 's/^aggregation-strategy = .*/aggregation-strategy = "fedavg"/' "$PYPROJECT"
sed -i 's/^seed = .*/seed = 42/' "$PYPROJECT"
echo ""
echo "Done. Results saved in results/"
