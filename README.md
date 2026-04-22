# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune itself during training using learnable gates and L1 regularization.

## Key Idea
Each weight has a gate (0 to 1):
- Gate ≈ 1 → weight is active
- Gate ≈ 0 → weight is removed (pruned)

## Method
- Custom PrunableLinear layer
- Gates learned using sigmoid
- L1 penalty encourages sparsity

## Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.0 | 100% | 44% |
| 0.05 | 12.4% | 100% |
| 0.5 | 12.4% | 100% |

## Trade-off
Higher lambda → more sparsity but lower accuracy

## Files
- self_pruning_neural_network.py → code
- self_pruning_report.pdf → explanation

## Note: Due to network timeout, synthetic data (FakeData) was used as fallback for demonstration.
  
- gate_distribution.png → gate analysis

## How to Run
1. Install PyTorch
2. Run:
   python self_pruning_neural_network.py
