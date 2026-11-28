# Octopus household-load backtest (research)

Rolling-origin backtest of TimesFM 3.0 (MLX, zero-shot, optional calendar covariates) vs seasonal-naive, moving-average, median and ridge baselines on a local SQLite of 30-min consumption. See `backtest.py` docstring; run with `--db` and `--out`. No data is committed; only aggregated metrics are written to `--out`. Results and reproduce notes live outside the repo (hub `timesfm-octopus/`).
