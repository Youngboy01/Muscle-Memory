import numpy as np


def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    """
    Update Elo ratings based on pairwise comparison results.

    Args:
        ratings: Dictionary mapping model names to their current Elo ratings
        matches: List of tuples (model_a, model_b, result) where result is 'a', 'b', or 'draw'
        k_factor: The K-factor controlling rating update magnitude

    Returns:
        Dictionary with updated ratings for all models
    """

    def expected_score(Ra, Rb):
        return 1 / (1 + 10 ** ((Rb - Ra) / 400))

    def update(Ra, Rb, Sa, K):
        Ea = expected_score(Ra, Rb)
        Eb = expected_score(Rb, Ra)
        Ra_mod = Ra + K * (Sa - Ea)
        Rb_mod = Rb + K * (1 - Sa - Eb)
        return Ra_mod, Rb_mod

    for m1, m2, winner in matches:
        Ra = ratings[m1]
        Rb = ratings[m2]
        if winner == "a":
            Sa = 1
        elif winner == "b":
            Sa = 0
        else:
            Sa = 0.5
        Ra_updated, Rb_updated = update(Ra, Rb, Sa, k_factor)
        ratings[m1] = Ra_updated
        ratings[m2] = Rb_updated

    return ratings

    pass
