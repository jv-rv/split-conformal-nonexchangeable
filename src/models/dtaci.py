"""Dynamically-tuned Adaptive Conformal Inference.

Implementation of 'Conformal Inference for Online Prediction with Arbitrary
Distribution Shifts', Gibbs and Candès (2023), https://arxiv.org/abs/2208.08401
"""

import numpy as np
from numpy.typing import NDArray


def loss(
    x: NDArray,
    alpha: float,
) -> NDArray:
    """Pinball loss used in DtACI."""
    return alpha * x - np.minimum(0, x)


def dtaci(
    betas: NDArray,
    gammas: NDArray,
    alpha: float,
    I: int,
) -> NDArray:
    """Select miscoverage levels in an online fashion according to DtACI.

    Args:
        betas: observed quantile levels that achieve smallest prediction sets
        gammas: step-size candidates
        alpha: target miscoverage level
        I: size of local time interval

    Returns:
        Sequence of miscoverage levels
    """
    T = len(betas)
    k = len(gammas)

    assert I <= T, "Local window cannot be larger than full window"

    denom = (1 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1 - alpha) ** 3
    eta = np.sqrt(3 / I) * np.sqrt((np.log(k * I) + 2) / denom)
    sigma = 1 / (2 * I)
    w = np.ones(k)
    alphas = np.full(k, alpha)
    alpha_agg = np.empty(T)

    for t in range(T):
        p = w / np.sum(w)
        alpha_agg[t] = (p * alphas).sum()
        w_bar = w * np.exp(-eta * loss(betas[t] - alphas, alpha))
        w = (1 - sigma) * w_bar / w_bar.sum() + sigma / k
        err = np.array(alphas > betas[t])
        alphas += gammas * (alpha - err)

    alpha_agg = np.clip(alpha_agg, 0, 1)

    return alpha_agg
