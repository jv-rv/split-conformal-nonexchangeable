"""Marginal coverage and empirical coverage evaluation."""

from argparse import ArgumentParser

import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from src.models import ConformalizedQR
from src.models.quantile_regressors import QuantileRegressor
from src.utils import eval
from src.utils.data import get_synthetic, SequentialSplit
from src.utils.dependence import minimize_eta_test_set
from src.utils.general import get_dir
from src.utils.model import get_model

parser = ArgumentParser()

# Set stochastic process as positional argument and initialize subparser
subparsers = parser.add_subparsers(dest="stochastic_process")

# Set parameters of each individual process depending on previous choice
parser_ar1 = subparsers.add_parser("ar1")
parser_ar1.add_argument("--phi", type=float)

parser_crw = subparsers.add_parser("cycle_random_walk")
parser_crw.add_argument("--prob_backward", "-b", type=float)
parser_crw.add_argument("--prob_stay", "-s", type=float)
parser_crw.add_argument("--prob_forward", "-f", type=float)
parser_crw.add_argument("--vertices", "-v", type=int)

parser_renewal = subparsers.add_parser("renewal")
parser_renewal.add_argument("--n_value", type=int)

parser_tsmc = subparsers.add_parser("two_state_markov_chain")
parser_tsmc.add_argument("--prob_p", "-p", type=float)
parser_tsmc.add_argument("--prob_q", "-q", type=float)

# Set common parameters
parser.add_argument("--quantile_model", type=str, default="boosting")
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_cal", type=int, default=1000)
parser.add_argument("--n_test", type=int, default=1)
parser.add_argument("--cal_before_train", action="store_true")
parser.add_argument("--lags", type=int, default=10)
parser.add_argument("--simulations", type=int, default=1000)
parser.add_argument("--n_jobs", type=int, default=-1)

args = parser.parse_args()


def get_stochastic_process_params(
    stochastic_process: str,
) -> tuple[dict[str, int | float], str]:
    """Retrieve stochastic process parameters."""
    match stochastic_process:
        case "ar1":
            params = {"phi": args.phi}
            suffix = f"-phi_{args.phi}"
        case "cycle_random_walk":
            params = {
                "b": args.prob_backward,
                "s": args.prob_stay,
                "f": args.prob_forward,
                "vertices": args.vertices,
            }
            suffix = (
                f"-b_{args.prob_backward}-s_{args.prob_stay}-f_{args.prob_forward}" +
                f"-v_{args.vertices}"
            )
        case "renewal":
            params = {"n": args.n_value}
            suffix = f"-n_{args.n_value}"
        case "two_state_markov_chain":
            params = {"p": args.prob_p, "q": args.prob_q}
            suffix = f"-p_{args.prob_p}-q_{args.prob_q}"
    return params, suffix


def run(
    i: int,
    params: dict[str, int | float],
    Model: QuantileRegressor,
    stochastic_process: str,
    quantile_model: str,
    alpha: float,
    n_train: int,
    n_cal: int,
    n_test: int,
    cal_before_train: bool,
    lags: int,
    simulations: int,
    n_jobs: int,
    **kwargs: int | float,
) -> dict[str, float]:
    """Generate random sequence with `i` as seed and run a single experiment."""
    df = get_synthetic(stochastic_process, N=n_train+n_cal+n_test, lags=lags, seed=i, **params)

    if not cal_before_train:
        train_index, cal_index, test_index = tuple(
            *SequentialSplit((n_train, n_cal, n_test), show_progress=False).split(df)
        )
    else:
        cal_index, train_index, test_index = tuple(
            *SequentialSplit((n_cal, n_train, n_test), show_progress=False).split(df)
        )

    # Split data
    X = df.drop("target", axis=1).to_numpy()
    y = df["target"].to_numpy()

    X_train, y_train = X[train_index], y[train_index]
    X_cal, y_cal = X[cal_index], y[cal_index]
    X_test, y_test = X[test_index], y[test_index]

    # Generate prediction intervals
    cqr = ConformalizedQR(Model, alpha, seed=0)
    cqr.fit(X_train, y_train)
    cqr.calibrate(X_cal, y_cal)
    y_pred_lower, y_pred_upper = cqr.predict(X_test)

    # Evaluate
    empirical_coverage = eval.empirical_coverage(y_test, y_pred_lower, y_pred_upper)
    average_interval_size = eval.average_interval_size(y_pred_lower, y_pred_upper)

    del df, train_index, cal_index, test_index, cqr
    del X, y, X_train, y_train, X_cal, y_cal, X_test, y_test, y_pred_lower, y_pred_upper

    if n_test == 1:
        return {
            "empirical_coverage": empirical_coverage,
            "average_interval_size": average_interval_size,
        }

    elif n_test > 1:
        delta_cal = 0.005
        delta_test = 0.005

        # Minimize eta for given process
        (
            eta,
            epsilon_cal,
            a_cal,
            m_cal,
            r,
            epsilon_test,
            a_test,
            m_test,
        ) = minimize_eta_test_set(
            n_cal=n_cal,
            n_test=n_test,
            alpha=alpha,
            delta_cal=delta_cal,
            delta_test=delta_test,
            p=args.prob_p,
            q=args.prob_q,
            bound="bernstein",
            stochastic_process=stochastic_process,
        )
        guarantee = (empirical_coverage >= (1 - alpha - eta))

        return {
            "empirical_coverage": empirical_coverage,
            "average_interval_size": average_interval_size,
            "delta_cal": delta_cal,
            "delta_test": delta_test,
            "eta": eta,
            "epsilon_cal": epsilon_cal,
            "a_cal": a_cal,
            "m_cal": m_cal,
            "r": r,
            "epsilon_test": epsilon_test,
            "a_test": a_test,
            "m_test": m_test,
            "guarantee": guarantee,
        }

    else:
        raise ValueError("At least 1 test point required.")


def main(
    stochastic_process: str,
    quantile_model: str,
    alpha: float,
    n_train: int,
    n_cal: int,
    n_test: int,
    cal_before_train: bool,
    lags: int,
    simulations: int,
    n_jobs: int,
    **kwargs: int | float,
) -> None:
    del kwargs
    params, suffix = get_stochastic_process_params(stochastic_process)
    Model = get_model(quantile_model)

    res = Parallel(n_jobs=n_jobs)(
        delayed(run)(i, params, Model, **vars(args)) for i in tqdm(range(simulations))
    )

    res = pd.DataFrame(res)

    res["model"] = quantile_model
    res["alpha"] = alpha
    res["n_train"] = n_train
    res["n_cal"] = n_cal
    res["n_test"] = n_test
    res["cal_before_train"] = cal_before_train
    res["n_lags"] = lags
    res["params"] = str(params)

    if n_test == 1:
        outpath = get_dir("eval/results/synthetic/marginal_coverage")
    elif n_test > 1:
        outpath = get_dir("eval/results/synthetic/empirical_coverage")

    cbt = "-cal_before_train" if cal_before_train else ""

    res.to_csv(
        f"{outpath}/{stochastic_process}{suffix}-{quantile_model}-alpha_{alpha}-" +
        f"n1_{n_train}-n2_{n_cal}-n3_{n_test}-lags_{lags}{cbt}.csv",
        index=False,
    )


if __name__ == "__main__":
    main(**vars(args))
