import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    n_samples = X.shape[0]
    
    for _ in range(n_iterations):
        if method == 'batch':
            y_pred = np.dot(X, weights)
            error = y_pred - y
            gradient = 2 * np.dot(X.T, error) / n_samples  
            weights -= learning_rate * gradient
            
        elif method =='stochastic':
            for i in range(n_samples):
                y_pred_i = np.dot(X[i], weights)
                error_i = y_pred_i - y[i]
                gradient_i = 2 * X[i] * error_i  
                weights -= learning_rate * gradient_i
                
        elif method =='mini_batch':
            for i in range(0, n_samples, batch_size):
                X_batch = X[i: i + batch_size]
                y_batch = y[i: i +batch_size]
                
                y_pred_batch = np.dot(X_batch, weights)
                error_batch = y_pred_batch - y_batch
                gradient_batch = 2 * np.dot(X_batch.T, error_batch) / batch_size
                weights -= learning_rate * gradient_batch
                
    return weights

if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    y = np.array([2, 3, 4, 5])

    # Parameters
    learning_rate = 0.01
    n_iterations = 1000
    batch_size = 2

    # Initialize weights
    weights = np.zeros(X.shape[1])

    # Test Batch Gradient Descent
    final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
    print(final_weights)
    # Test Stochastic Gradient Descent
    final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
    print(final_weights)
    # Test Mini-Batch Gradient Descent
    final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch')
    print(final_weights)