import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0
    # Your code here
    for _ in range(max_iter):
        pred = np.dot(X, weights) + bias
        error = pred - y
        gradien_w = np.dot(X.T, error) / n_samples + alpha * np.sign(weights)
        gradien_b = np.sum(error) / n_samples

        weights -= learning_rate * gradien_w
        bias -=  learning_rate * gradien_b

        if np.linalg.norm(gradien_w, ord=1) < tol:
            break

    return weights, bias
    
if __name__ == "__main__":
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 2])

    alpha = 0.1
    weights, bias = l1_regularization_gradient_descent(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)
    print(weights, bias)