#!/usr/bin/env bash
# Run federated learning with each aggregation strategy, keeping everything
# else fixed (MonteCarloSampler, same seed, same data splits).

set -euo pipefail

PYPROJECT="pyproject.toml"
STRATEGIES=("fedavg" "fedadam" "fedadagrad" "fedavgm")

# Ensure sampler is monte-carlo
sed -i 's/^sampler-type = .*/sampler-type = "monte-carlo"/' "$PYPROJECT"

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "=========================================="
    echo " Running with aggregation-strategy: $strategy"
    echo "=========================================="
    echo ""

    # Update aggregation-strategy in pyproject.toml
    sed -i "s/^aggregation-strategy = .*/aggregation-strategy = \"$strategy\"/" "$PYPROJECT"

    flwr run .
done

# Reset to default
sed -i 's/^aggregation-strategy = .*/aggregation-strategy = "fedavg"/' "$PYPROJECT"
echo ""
echo "Done. Results saved in results/"
