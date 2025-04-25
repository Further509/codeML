import math
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	weights = np.array(weights)
	labels = np.array(labels)
	logits = np.dot(features, weights) + bias
	probabilities = sigmoid(logits)
	mse = np.mean((probabilities - labels) ** 2)
	probabilities = np.round(probabilities, 4)
	mse = np.round(mse, 4)
	return probabilities.tolist(), mse.tolist()

if __name__ == "__main__":
	features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]] 
	labels = [0, 1, 0] 
	weights = [0.7, -0.4] 
	bias = -0.1
	ans = single_neuron_model(features, labels, weights, bias)
	print(ans)