#!/bin/bash

# Create logs directory if not exists
mkdir -p logs

echo "Launching DQN Walk-Forward campaigns..."

# GPU 0 Tasks (Sequential execution)
echo "Launching GPU 0 tasks (Reliance portfolio_return, Reliance diff_sortino, TCS portfolio_return)..."
(
  echo "=== RELIANCE PORTFOLIO_RETURN ==="
  CUDA_VISIBLE_DEVICES=0 python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_2 --reward-type portfolio_return --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
  
  echo "=== RELIANCE DIFF_SORTINO ==="
  CUDA_VISIBLE_DEVICES=0 python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_2 --reward-type diff_sortino --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
  
  echo "=== TCS PORTFOLIO_RETURN ==="
  CUDA_VISIBLE_DEVICES=0 python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_2 --reward-type portfolio_return --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
) > logs/dqn_gpu0.log 2>&1 &

# GPU 1 Tasks (Sequential execution)
echo "Launching GPU 1 tasks (TCS diff_sortino, HDFCBANK portfolio_return, HDFCBANK diff_sortino)..."
(
  echo "=== TCS DIFF_SORTINO ==="
  CUDA_VISIBLE_DEVICES=1 python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_2 --reward-type diff_sortino --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
  
  echo "=== HDFCBANK PORTFOLIO_RETURN ==="
  CUDA_VISIBLE_DEVICES=1 python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_2 --reward-type portfolio_return --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
  
  echo "=== HDFCBANK DIFF_SORTINO ==="
  CUDA_VISIBLE_DEVICES=1 python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_2 --reward-type diff_sortino --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
) > logs/dqn_gpu1.log 2>&1 &

# GPU 2 Tasks (Sequential execution)
echo "Launching GPU 2 tasks (INFY portfolio_return, INFY diff_sortino)..."
(
  echo "=== INFY PORTFOLIO_RETURN ==="
  CUDA_VISIBLE_DEVICES=2 python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_2 --reward-type portfolio_return --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
  
  echo "=== INFY DIFF_SORTINO ==="
  CUDA_VISIBLE_DEVICES=2 python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_2 --reward-type diff_sortino --action-space-type discrete_3 --algo dqn --config configs/dqn_best_params.yaml
) > logs/dqn_gpu2.log 2>&1 &

echo "All DQN scaling tasks have been launched in the background!"
echo "Check GPU 0 log: tail -f logs/dqn_gpu0.log"
echo "Check GPU 1 log: tail -f logs/dqn_gpu1.log"
echo "Check GPU 2 log: tail -f logs/dqn_gpu2.log"
