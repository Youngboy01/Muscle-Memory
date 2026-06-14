import numpy as np


def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.

    Args:
        log_probs: List of lists, where each inner list contains
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)

    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    # def conversion(log_probs):
    #     stab = log_probs - np.max(log_probs)
    #     new = np.exp(stab)
    #     return new/np.sum(new)
    log_probs = np.array(log_probs)
    correct_ans = np.array(correct_answers)
    m = len(correct_ans)
    preds = np.argmax(log_probs, axis=1)
    accuracy = np.mean(preds == correct_ans)
    # pred = []
    # for i in range(m):
    #     maxi = np.argmax(log_probs[i])
    #     pred.append(maxi)
    # cnt = 0
    # for i in range(m):
    #     if pred[i]==correct_answers[i]:
    #         cnt+=1
    # accuracy = cnt/(len(correct_answers))
    stable = log_probs - np.max(log_probs, axis=1, keepdims=True)
    exp = np.exp(stable)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    row_idx = np.arange(m)
    correct_probs = probs[row_idx, correct_ans]
    avg_correct_prob = np.mean(correct_probs)

    # correct_prob = 0
    # for i in range(m):
    #     converted = conversion(log_probs[i])

    #     corr_idx = correct_answers[i]
    #     correct_prob += converted[corr_idx]
    # avg_correct_prob = correct_prob/m

    return {
        "accuracy": accuracy,
        "predictions": preds.tolist(),
        "avg_correct_prob": round(avg_correct_prob, 4),
    }

    pass
