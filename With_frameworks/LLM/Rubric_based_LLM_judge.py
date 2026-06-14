import numpy as np


def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0,
) -> dict:
    """
    Evaluate LLM response using rubric-based multi-judge scoring.

    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion

    Returns:
        Dictionary with evaluation results
    """
    m = len(judge_scores)
    n = len(judge_scores[0])
    judges = np.array(judge_scores)
    weights = np.array(criteria_weights)

    criterion_scores = np.round(judges.mean(axis=0), 4)
    weighted_score = np.dot(weights, criterion_scores)
    normalized = weighted_score / max_score

    passed = False
    if normalized >= passing_threshold:
        passed = True

    std_dev = np.std(judges, axis=0)
    mean_std_dev = std_dev.mean(axis=0)
    std_max = max_score / 2
    agreement = 1 - (mean_std_dev / std_max)

    return {
        "weighted_score": round(weighted_score, 4),
        "normalized_score": round(normalized, 4),
        "criterion_scores": criterion_scores.tolist(),
        "pass_status": passed,
        "judge_agreement": round(agreement, 4),
    }

    pass
