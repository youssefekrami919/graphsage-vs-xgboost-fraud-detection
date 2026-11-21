#!/usr/bin/env bash
# ==============================================================================
# One-Command Fully Reproducible Benchmark Execution Script
# Paper Title: A GraphSAGE Approach for Fraud Detection on the IEEE-CIS Dataset:
#              A Comparative Study with XGBoost on Accuracy and Inference Latency
# ==============================================================================

set -e

echo "=== [1/3] Environment Verification ==="
python -c "import torch, xgboost, torch_geometric; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | XGBoost: {xgboost.__version__}')"

echo "=== [2/3] Running Rapid Prototype (100k Transactions) ==="
python main.py --mode prototype --seed 42

echo "=== [3/3] Execution Complete ==="
echo "Tables exported to: results/prototype/tables/"
echo "Figures exported to: results/prototype/figures/"
echo "Models saved to: models/prototype/native/ and models/prototype/onnx/"
