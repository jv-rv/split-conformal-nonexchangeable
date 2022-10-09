"""General functions for data dependence analysis."""


import numpy as np
from numpy.linalg import matrix_power
from numpy.typing import NDArray


def beta_two_state_markov_chain(
    p: float,
    q: float,
    t: int | NDArray,
) -> NDArray:
    """Calculate beta-mixing coefficients of a two-state markov chain.

    Args:
        p: probability of going from state 0 to state 1.
        q: probability of going from state 1 to state 0.
        t: separation parameters.

    Returns:
        Beta-mixing coefficients.
    """
    return np.abs(1 - p - q)**t * (2 * p * q) / (p + q)**2


def beta_markov_chain(
    P: NDArray,
    pi: NDArray,
    t: int,
) -> float:
    """Calculate beta-mixing coefficient of a markov chain.

    Args:
        P: transition matrix.
        pi: stationary distribution.
        t: separation parameter.

    Returns:
        Beta-mixing coefficient.
    """
    assert t > 0, "Beta-mixing coefficient is defined only for strictly positive integers."
    Pt = matrix_power(P, t)
    beta = 0.5 * pi @ np.abs(Pt - pi).sum(axis=1)
    beta = beta.item()
    assert isinstance(beta, float)
    return beta


def minimize_eta_test_point(
    n_cal: int,
    alpha: float,
    delta_cal: float,
    p: float,
    q: float,
    bound: str,
    stochastic_process: str,
) -> tuple[float, float, int, int, int]:
    """Minimize dependence factor `eta` for marginal coverage.

    Args:
        n_cal: calibration set size.
        alpha: miscoverage level.
        delta_cal: one minus probability of result holding.
        p: probability of going from state 0 to state 1 in Markov chain.
        q: probability of going from state 1 to state 0 in Markov chain.
        bound: inequality used to bound `eta`.
        stochastic_process: process that originated the data sequence.

    Returns:
        Best bound achieved after optimization procedure.
        Minimum dependence correction factor `eta`.
        Block size `a` that minimizes `eta`.
        Half the amount of consecutive blocks `m` that minimizes `eta`.
        Distance between training and calibration `r` that minimizes `eta`.
    """
    if stochastic_process == "two_state_markov_chain":
        beta = beta_two_state_markov_chain
    else:
        raise ValueError(
            f"Beta calculation for stochastic process {stochastic_process} is not supported.",
        )
    feasible = np.array([(n_cal // 2) // k for k in range(1, (n_cal // 2) + 1)])

    M = np.concatenate([np.arange(1, v + 1) for v in feasible])
    A = np.repeat(np.arange(1, (n_cal // 2) + 1), feasible)
    R = n_cal - 2 * M * A + 1

    positive_log = (
        delta_cal > 4 * (M - 1) * beta(p, q, A) + beta(p, q, R)
    )
    positive_distance = (R >= 1)

    M = M[positive_log & positive_distance]
    A = A[positive_log & positive_distance]
    R = R[positive_log & positive_distance]

    if bound == "bernstein":
        BETA_ARRAY = np.tile(
            beta(p, q, np.arange(1, max(A) + 1)),
            reps=(len(A), 1),
        )
        MUL_FACTOR_ARRAY = np.maximum(
            0,
            np.tile(
                (A - 1).reshape(-1, 1), reps=(1, max(A)),
            ) - np.arange(max(A)),
        )
        BETA_SUM = (BETA_ARRAY * MUL_FACTOR_ARRAY).sum(axis=1)

        LOG_TERM = np.log(
            4 / (
                delta_cal - 4 * (M - 1) * beta(p, q, A) - beta(p, q, R)
            ),
        )
        VARIANCE_TERM = (1 - alpha) * alpha + (2 / A) * BETA_SUM
        EPSILON_CAL = (
            np.sqrt(VARIANCE_TERM * 4 / (n_cal - R + 1) * LOG_TERM) +
            1 / (3 * M) * LOG_TERM + (R - 1) / n_cal
        )
    EPSILON_TRAIN = beta(p, q, n_cal + 1)
    eta = EPSILON_CAL + EPSILON_TRAIN + delta_cal
    objective = 1 - alpha - eta
    idx = objective.argmax()
    return objective[idx], eta[idx], A[idx], M[idx], R[idx]


def minimize_eta_test_set(
    n_cal: int,
    n_test: int,
    alpha: float,
    delta_cal: float,
    delta_test: float,
    p: float,
    q: float,
    bound: str,
    stochastic_process: str,
) -> tuple[float, float, int, int, int, float, int, int]:
    """Minimize dependence factor `eta` for empirical coverage.

    Args:
        n_cal: calibration set size.
        n_test: test set size.
        alpha: miscoverage level.
        delta_cal: one minus `delta_test` minus probability of result holding.
        delta_test: one minus `delta_cal` minus probability of result holding.
        p: probability of going from state 0 to state 1 in Markov chain.
        q: probability of going from state 1 to state 0 in Markov chain.
        bound: inequality used to bound `eta`.
        stochastic_process: process that originated data sequence.

    Returns:
        Minimum dependence correction factor `eta`.
        Minimum dependence correction factor `epsilon_cal`.
        Block size `a` that minimizes `epsilon_cal`.
        Half the amount of consecutive blocks `m` that minimizes `epsilon_cal`.
        Distance between training and calibration `r` that minimizes `epsilon_cal`.
        Minimum dependence correction factor `epsilon_test`.
        Block size `a` that minimizes `epsilon_test`.
        Half the amount of consecutive blocks `m` that minimizes `epsilon_test`.
    """
    if stochastic_process == "two_state_markov_chain":
        beta = beta_two_state_markov_chain
    else:
        raise ValueError(
            f"Beta calculation for stochastic process {stochastic_process} is not supported.",
        )

    # Refrain from dealing with odd calibration or test sets for simplicity
    assert n_cal % 2 == 0
    assert n_test % 2 == 0

    # Possible values on calibration set
    feasible_cal = np.array([(n_cal // 2) // k for k in range(1, (n_cal // 2) + 1)])

    M_CAL = np.concatenate([np.arange(1, v + 1) for v in feasible_cal])
    A_CAL = np.repeat(np.arange(1, (n_cal // 2) + 1), feasible_cal)
    R = n_cal - 2 * M_CAL * A_CAL + 1

    positive_log_cal = (
        delta_cal > 4 * (M_CAL - 1) * beta(p, q, A_CAL) + beta(p, q, R)
    )
    positive_distance = (R >= 1)

    M_CAL = M_CAL[positive_log_cal & positive_distance]
    A_CAL = A_CAL[positive_log_cal & positive_distance]
    R = R[positive_log_cal & positive_distance]

    # Possible values on test set
    A_TEST = np.arange(1, (n_test // 2) + 1)
    M_TEST = np.array([(n_test // 2) // k for k in A_TEST])
    constraint_test = (2 * M_TEST * A_TEST == n_test)

    A_TEST = A_TEST[constraint_test]
    M_TEST = M_TEST[constraint_test]

    positive_log_test = (
        delta_test > 4 * (M_TEST - 1) * beta(p, q, A_TEST) + beta(p, q, n_cal)
    )

    M_TEST = M_TEST[positive_log_test]
    A_TEST = A_TEST[positive_log_test]

    if bound == "bernstein":
        # Calibration factors
        BETA_CAL_ARRAY = np.tile(
            beta(p, q, np.arange(1, max(A_CAL) + 1)).reshape(1, -1),
            reps=(len(A_CAL), 1),
        )
        MUL_FACTOR_CAL_ARRAY = np.maximum(
            0,
            np.tile(
                (A_CAL - 1).reshape(-1, 1), reps=(1, max(A_CAL)),
            ) - np.arange(max(A_CAL)),
        )
        BETA_SUM_CAL = (BETA_CAL_ARRAY * MUL_FACTOR_CAL_ARRAY).sum(axis=1)

        LOG_TERM_CAL = np.log(
            4 / (delta_cal - 4 * (M_CAL - 1) * beta(p, q, A_CAL) - beta(p, q, R)),
        )
        VARIANCE_TERM = (1 - alpha) * alpha + (2 / A_CAL) * BETA_SUM_CAL
        EPSILON_CAL = (
            np.sqrt(VARIANCE_TERM * 4 / (n_cal - R + 1) * LOG_TERM_CAL) +
            1 / (3 * M_CAL) * LOG_TERM_CAL + (R - 1) / n_cal
        )

        # Test factors
        BETA_TEST_ARRAY = np.tile(
            beta(p, q, np.arange(1, max(A_TEST) + 1)).reshape(1, -1),
            reps=(len(A_TEST), 1),
        )
        MUL_FACTOR_TEST_ARRAY = np.maximum(
            0,
            np.tile(
                (A_TEST - 1).reshape(-1, 1), reps=(1, max(A_TEST)),
            ) - np.arange(max(A_TEST)),
        )
        BETA_SUM_TEST = (BETA_TEST_ARRAY * MUL_FACTOR_TEST_ARRAY).sum(axis=1)

        LOG_TERM_TEST = np.log(
            4 / (delta_test - 4 * (M_TEST - 1) * beta(p, q, A_TEST) - beta(p, q, n_cal)),
        )
        VARIANCE_TERM = (1 - alpha) * alpha + (2 / A_TEST) * BETA_SUM_TEST
        EPSILON_TEST = (
            np.sqrt(VARIANCE_TERM * 4 / n_test * LOG_TERM_TEST) +
            1 / (3 * M_TEST) * LOG_TERM_TEST
        )

    idx_cal = EPSILON_CAL.argmin()
    idx_test = EPSILON_TEST.argmin()

    eta = EPSILON_CAL[idx_cal] + EPSILON_TEST[idx_test]

    return (
        eta,
        EPSILON_CAL[idx_cal],
        A_CAL[idx_cal],
        M_CAL[idx_cal],
        R[idx_cal],
        EPSILON_TEST[idx_test],
        A_TEST[idx_test],
        M_TEST[idx_test],
    )
