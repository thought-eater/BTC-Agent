# Deadline Report (20h mode)

Generated: 2026-02-27T06:47:25.576443Z
Run ID: run_20260227_013925
Deadline UTC: 2026-02-27T11:39:25.072610Z
Freeze policy: graceful

## Runtime
- total_budget_hours: 10
- deadline_hours: 10.0
- gpu_mode_selected: single
- parallel_workers: 1

## Stage Results
- prep: {"stage": "prep", "elapsed_sec": 49.188614041078836, "budget_sec": 1800.0, "stopped_reason": "completed"}
- trade_pred: {"stage": "trade", "elapsed_sec": 6184.111236994155, "budget_sec": 5400.0, "stopped_reason": "completed"}
- main: {"stage": "main", "elapsed_sec": 12245.510543046985, "budget_sec": 21600.0, "stopped_reason": "completed"}
- eval: {"stage": "eval", "elapsed_sec": 1.2065272717736661, "budget_sec": 1800.0, "stopped_reason": "completed"}

## Job Summary
- prep: status=completed variant=n/a method=n/a
- trade: status=completed variant=paper method=n/a
- pred_paper: status=completed variant=paper method=n/a
- main_classic_paper_a30_o16: status=completed variant=paper method=classic
- main_proposed_paper_a30_o16: status=completed variant=paper method=proposed
- main_proposed_policy_gradient_a30_o16: status=completed variant=policy_gradient method=proposed
- main_classic_paper_a55_o16: status=completed variant=paper method=classic
- main_proposed_paper_a55_o16: status=completed variant=paper method=proposed
- main_proposed_policy_gradient_a55_o16: status=completed variant=policy_gradient method=proposed
- main_classic_paper_a80_o16: status=completed variant=paper method=classic
- main_proposed_paper_a80_o16: status=completed variant=paper method=proposed
- main_proposed_policy_gradient_a80_o16: status=completed variant=policy_gradient method=proposed

## Artifacts
- data/processed/btc_price_clean.csv
- data/processed/tweet_sentiment_scores.csv
- data/processed/integrated_base.csv
- data/processed/x1_trade_recommendations.csv
- data/processed/x2_price_predictions.csv
- data/processed/integrated_dataset.csv
- /home/grupo1/BTCAgent/results/20260227_001839/table4_risk30.csv
- /home/grupo1/BTCAgent/results/20260227_001839/table5_risk55.csv
- /home/grupo1/BTCAgent/results/20260227_001839/table6_risk80.csv
- /home/grupo1/BTCAgent/results/20260227_001839/table_improvement_delta.csv
- /home/grupo1/BTCAgent/results/20260227_001839/table_improvement_summary.csv
- /home/grupo1/BTCAgent/results/20260227_001839/table8_sota_comparison.csv
- /home/grupo1/BTCAgent/results/20260227_001839/price_history.png
- /home/grupo1/BTCAgent/results/20260227_001839/trading_signals.png
- /home/grupo1/BTCAgent/results/20260227_001839/improvement_comparison.png

## Validation
- min_improvements_required: 1 
- improvements_completed: 1 
- requirement_met: True 