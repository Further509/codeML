import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    standardized_data = (data - mu) / sigma
    mini = np.min(data, axis=0)
    maxm = np.max(data, axis=0)
    normalized_data = (data - mini) / (maxm - mini)
    return np.round(standardized_data, 4), np.round(normalized_data, 4)

if __name__ == "__main__":
    data = np.array([[1, 2], [3, 4], [5, 6]])
    standardized_data, normalized_data = feature_scaling(data)
    print("Standardized Data:\n", standardized_data)
    print("Normalized Data:\n", normalized_data)
    print(np.min(data, axis=1).shape, np.min(data, axis=0).shape)