"""Utilities to deal with data."""

import warnings

import numpy as np
import pandas as pd

from src.utils.stochastic_processes import AR1, CycleRandomWalk, Renewal, TwoStateMarkovChain


def get_data(
    target: str,
    target_gap: int,
    maxlags: int,
    year: int,
) -> pd.DataFrame:
    """Get processed data with specified lagged features and set target variable."""
    if target and target_gap < 1:
        raise ValueError("A gap less than 1 would constitute data leakage and is not allowed.")
    if not target and target_gap:
        warnings.warn(
            "Warning: No target variable was passed, but a gap was given. Gap will be ignored.",
        )
    df = pd.read_csv(
        f"data/processed/df-{target}-{year}-maxlags_{maxlags}.csv",
        index_col="datetime",
        parse_dates=["datetime"],
    )
    if target:
        df["target"] = df[target].shift(-target_gap)
    return df.dropna(axis=0)


def get_synthetic(
    stochastic_process: str,
    N: int,
    lags: int,
    seed: int | None = None,
    **kwargs: int | float,
) -> pd.DataFrame:
    """Generate dataset from stochastic process."""
    # Initialize stochastic process class
    if stochastic_process == "ar1":
        sp = AR1(**kwargs)
    elif stochastic_process == "cycle_random_walk":
        sp = CycleRandomWalk(**kwargs)
    elif stochastic_process == "renewal":
        sp = Renewal(**kwargs)
    elif stochastic_process == "two_state_markov_chain":
        sp = TwoStateMarkovChain(**kwargs)
    else:
        raise ValueError(f"Stochastic process {stochastic_process} is not available.")

    # Generate sequence
    sequence = sp.generate(N=N+lags+1, seed=seed)

    # Add small gaussian noise to discrete sequences.
    # This is particularly important for binary sequences, otherwise
    # it would not be possible to compute meaningful quantiles.
    match stochastic_process:
        case "cycle_random_walk" | "renewal" | "two_state_markov_chain":
            rng = np.random.default_rng(seed)
            sequence = rng.normal(sequence, scale=1e-6)

    # Create dataframe with original sequence and lags
    df = pd.DataFrame(sequence, columns=["value"])
    for lag in range(1, lags + 1):
        df[f"value_lag_{lag}"] = df["value"].shift(lag)

    # Set target as next unseen observation
    df["target"] = df["value"].shift(-1)

    # Drop missing values introduced during lagged variables creation
    df = df.dropna().reset_index(drop=True)

    # Verify resulting dataframe is of expected length
    assert len(df) == N

    return df
