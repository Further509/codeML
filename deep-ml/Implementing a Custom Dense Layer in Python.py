
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

    def set_input_shape(self, shape):

        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        input_size = self.input_shape[0]
        self.W = np.random.uniform(-1 / math.sqrt(input_size), 1 / math.sqrt(input_size), (input_size, self.n_units))
        self.w0 = np.zeros((1, self.n_units))

        self.optimizer_W = copy.deepcopy(optimizer)
        self.optimizer_w0 = copy.deepcopy(optimizer)

    def parameters(self):
        return self.W.size + self.w0.size

    def forward_pass(self, X, training='True'):
        self.layer_input = X
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        dX = np.dot(accum_grad, self.W.T)
        if self.trainable:
            dW = np.dot(self.layer_input.T, accum_grad)
            dw0 = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.optimizer_W.update(self.W, dW)
            self.w0 = self.optimizer_w0.update(self.w0, dw0)
        return dX

    def number_of_parameters(self):
        return self.parameters()
    
    def output_shape(self):
        return (self.n_units,)

if __name__ == "__main__":
    # Initialize a Dense layer with 3 neurons and input shape (2,)
    dense_layer = Dense(n_units=3, input_shape=(2,))

    # Define a mock optimizer with a simple update rule
    class MockOptimizer:
        def update(self, weights, grad):
            return weights - 0.01 * grad

    optimizer = MockOptimizer()

    # Initialize the Dense layer with the mock optimizer
    dense_layer.initialize(optimizer)

    # Perform a forward pass with sample input data
    X = np.array([[1, 2]])
    output = dense_layer.forward_pass(X)
    print("Forward pass output:", output)

    # Perform a backward pass with sample gradient
    accum_grad = np.array([[0.1, 0.2, 0.3]])
    back_output = dense_layer.backward_pass(accum_grad)
    print("Backward pass output:", back_output)