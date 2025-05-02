import numpy as np

def r_squared(y_true, y_pred):
    # Write your code here
    y_true_mean = np.mean(y_true)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true_mean) ** 2)
    return round(1 - sse / sst, 3) if sst != 0 else 0.0

if __name__ == "__main__":
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    print(r_squared(y_true, y_pred))
