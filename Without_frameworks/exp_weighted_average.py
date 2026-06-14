def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    # Your code here
    k = len(rewards)
    init_reward = ((1-alpha)**k)*Q1
    recency_weightage=0
    for i in range(k):
        recency_weightage += alpha*(1-alpha)**(k-i-1)*rewards[i]
    return init_reward + recency_weightage
    pass
