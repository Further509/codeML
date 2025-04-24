import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []

    for _ in range(epochs):
        logits = np.dot(features, weights) + bias
        predictions = sigmoid(logits)

        error = predictions - labels 
        mse = np.mean(error ** 2)
        mse_values.append(np.round(mse, 4))

        d_mse_d_pred = 2 * error / len(labels)
        d_pred_d_logits = sigmoid_derivative(logits)
        d_logits_d_weights = features
        d_logits_d_bias = 1

        d_mse_d_weights = np.dot(d_logits_d_weights.T, d_mse_d_pred * d_pred_d_logits)
        d_mse_d_bias = np.sum(d_mse_d_pred * d_pred_d_logits)

        weights -= learning_rate * d_mse_d_weights
        bias -= learning_rate * d_mse_d_bias

    updated_weights = np.round(weights, 4)
    updated_bias = np.round(bias, 4)
    return updated_weights, updated_bias, mse_values

if __name__ == "__main__":
    features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = np.array([1, 0, 0]) 
    initial_weights = np.array([0.1, -0.2])
    initial_bias = 0.0 
    learning_rate = 0.1 
    epochs = 2
    out = train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs)
    print(out)