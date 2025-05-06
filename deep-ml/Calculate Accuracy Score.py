import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    right = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = right / total
    return accuracy

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)
    print(output)