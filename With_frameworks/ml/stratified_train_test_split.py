import numpy as np


def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    train_idx, test_idx = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]

        np.random.shuffle(idx)

        split = int(len(idx) * test_size)

        train_idx.extend(idx[split:])
        test_idx.extend(idx[:split])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test
