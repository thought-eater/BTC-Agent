# Deadline Report (20h mode)

Generated: 2026-02-25T21:06:08.834055Z
Run ID: run_20260225_204749
Deadline UTC: 2026-02-25T21:05:49.511667Z
Freeze policy: graceful

## Runtime
- total_budget_hours: 24
- deadline_hours: 0.3
- gpu_mode_selected: single
- parallel_workers: 1

## Stage Results
- prep: {"stage": "prep", "elapsed_sec": 0.00024345982819795609, "budget_sec": 7200.0, "stopped_reason": "completed"}
- trade_pred: {"stage": "trade", "elapsed_sec": 747.5426612640731, "budget_sec": 10800.0, "stopped_reason": "completed"}
- main: {"stage": "main", "elapsed_sec": 350.01679476024583, "budget_sec": 36000.0, "stopped_reason": "completed"}
- eval: {"stage": "eval", "elapsed_sec": 1.0321164629422128, "budget_sec": 14400.0, "stopped_reason": "completed"}

## Job Summary
- prep: status=skipped variant=n/a method=n/a
- trade: status=completed variant=paper method=n/a
- pred_paper: status=completed variant=paper method=n/a
- pred_continuous: status=completed variant=continuous method=n/a
- main_classic_paper_a30_o8: status=completed variant=paper method=classic
- main_proposed_paper_a30_o8: status=completed variant=paper method=proposed
- main_proposed_dueling_double_a30_o8: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a30_o8: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a30_o16: status=frozen variant=paper method=classic
- main_proposed_paper_a30_o16: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a30_o16: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a30_o16: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a30_o24: status=frozen variant=paper method=classic
- main_proposed_paper_a30_o24: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a30_o24: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a30_o24: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a55_o8: status=frozen variant=paper method=classic
- main_proposed_paper_a55_o8: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a55_o8: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a55_o8: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a55_o16: status=frozen variant=paper method=classic
- main_proposed_paper_a55_o16: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a55_o16: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a55_o16: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a55_o24: status=frozen variant=paper method=classic
- main_proposed_paper_a55_o24: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a55_o24: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a55_o24: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a80_o8: status=frozen variant=paper method=classic
- main_proposed_paper_a80_o8: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a80_o8: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a80_o8: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a80_o16: status=frozen variant=paper method=classic
- main_proposed_paper_a80_o16: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a80_o16: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a80_o16: status=frozen variant=predictive_continuous method=proposed
- main_classic_paper_a80_o24: status=frozen variant=paper method=classic
- main_proposed_paper_a80_o24: status=frozen variant=paper method=proposed
- main_proposed_dueling_double_a80_o24: status=frozen variant=dueling_double method=proposed
- main_proposed_predictive_continuous_a80_o24: status=frozen variant=predictive_continuous method=proposed

## Artifacts
- data/processed/btc_price_clean.csv
- data/processed/tweet_sentiment_scores.csv
- data/processed/integrated_base.csv
- data/processed/x1_trade_recommendations.csv
- data/processed/x2_price_predictions.csv
- data/processed/x2_price_predictions_continuous.csv
- data/processed/integrated_dataset.csv
- data/processed/integrated_dataset_predictive_continuous.csv
- results/table4_risk30.csv
- results/table8_sota_comparison.csv
- results/price_history.png
- results/trading_signals.png

## Validation
- min_improvements_required: 2 
- improvements_completed: 0 
- requirement_met: False 