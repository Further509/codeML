import numpy as np
def precision(y_true, y_pred):
	# Your code here
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    
if __name__ == "__main__":
    # Example usage
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    print(precision(y_true, y_pred)) # Output: 1.0