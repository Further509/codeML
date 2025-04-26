import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    true_negetive = np.sum((y_true == 0) & (y_pred == 0))
    false_negetive = np.sum((y_true == 1) & (y_pred == 0))
    P = true_positive / (true_positive + false_positive)
    R = true_positive / (true_positive + false_negetive)
    return round((1 + beta ** 2) * P * R / (beta ** 2 * P + R), 3)

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    beta = 2
    print(f"F-Score: {f_score(y_true, y_pred, beta):.3f}")

    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    beta = 1

    print(f_score(y_true, y_pred, beta))

    y_true = np.array([1, 0, 1, 1, 0, 0]) 
    y_pred = np.array([1, 0, 1, 1, 0, 0]) 
    beta = 2 
    print(f_score(y_true, y_pred, beta))