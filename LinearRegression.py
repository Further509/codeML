import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(m, 1)
    for _ in range(iterations):
        h = np.dot(X, theta)
        error = h - y
        theta -= alpha / m * np.dot(X.T, error)
    return np.round(theta.flatten(), 4)

if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.01
    iterations = 1000
    theta = linear_regression_gradient_descent(X, y, alpha, iterations)
    print(theta)