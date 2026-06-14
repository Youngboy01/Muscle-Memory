import numpy as np
def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    # Your code here
    criteria = list(criteria_weights.keys())
    weights = np.array([criteria_weights[c] for c in criteria])
    weights = weights/weights.sum()
    tot_margin = 0.0
    ties, a_win, b_win = 0,0,0
    results = []
    for cm in comparisons:
        a = np.array([cm['scores_a'].get(c,0) for c in criteria])
        b = np.array([cm['scores_b'].get(c,0) for c in criteria])

        wt_a = np.dot(weights,a) 
        wt_b = np.dot(weights,b) 

        diff  = (wt_a - wt_b)
        tot_margin += abs(wt_a - wt_b)

        if abs(diff)<= tie_threshold:
            winner = 'tie'
            ties+=1    
        elif diff>0:
            winner = 'A'
            a_win +=1
        else:
            winner = 'B'
            b_win +=1
        
        results.append({
            'id': cm['id'],
            'winner': winner,
            'margin': round(abs(wt_a - wt_b), 4)
        })
    n = len(comparisons)
    if n == 0:
        return {
            'results': [],
            'win_rate_a': 0.0,
            'win_rate_b': 0.0,
            'tie_rate': 0.0,
            'avg_margin': 0.0
        }
    return {
        'results':results,
        'win_rate_a': round(a_win/n, 4),
        'win_rate_b': round(b_win/n,4),
        'tie_rate': round(ties/n,4),
        'avg_margin': round(tot_margin/n , 4) ,
    }