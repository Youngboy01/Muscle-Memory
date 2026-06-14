import numpy as np
from typing import List, Tuple


def fit_bradley_terry(
    comparisons: List[Tuple[int, int]],
    n_items: int,
    learning_rate: float = 0.5,
    n_iterations: int = 100,
) -> np.ndarray:
    """
    Fit Bradley-Terry model parameters using maximum likelihood estimation.

    Args:
        comparisons: List of (winner_idx, loser_idx) tuples
        n_items: Total number of items to rank
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations

    Returns:
        np.ndarray: Estimated strength parameters of shape (n_items,)
    """

    # Your code here
    def sigmoid(x):
        ans = np.where(x >= 0, 1 / (1 + np.exp(-x)), (np.exp(x) / (1 + np.exp(x))))
        return ans

    beta = np.zeros(n_items)
    for _ in range(n_iterations):
        grad = np.zeros(n_items)
        for winner, loser in comparisons:
            diff = beta[winner] - beta[loser]
            pij = sigmoid(diff)
            g = 1 - pij

            grad[winner] += g
            grad[loser] -= g

        beta += learning_rate * grad
        beta -= np.mean(beta, axis=0)

        # center = sum(beta)/(n_items)
        # for i in range(n_items):
        #     beta[i]-=center

    return beta
    pass
