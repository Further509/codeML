
import numpy as np
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.W_hy = np.random.randn(output_size, hidden_size)*0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        """
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        self.last_inputs = []
        self.last_hiddens = [h]

        for t in range(len(x)):
            self.last_inputs.append(x[t].reshape(-1, 1))
            h = np.tanh(np.dot(self.W_xh, self.last_inputs[t]) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            outputs.append(y)
            self.last_hiddens.append(h)

        self.last_outputs = outputs
        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(x))):
            dy = self.last_outputs[t] - y[t].reshape(-1, 1)
            dW_hy += np.dot(dy, self.last_hiddens[t + 1].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - self.last_hiddens[t+1] ** 2) * dh  # Derivative of tanh

            dW_xh += np.dot(dh_raw, self.last_inputs[t].T)
            dW_hh += np.dot(dh_raw, self.last_hiddens[t].T)
            db_h += dh_raw

            dh_next = np.dot(self.W_hh.T, dh_raw)

        # Update weights and biases
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

if __name__ == "__main__":
    np.random.seed(42)
    input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
    expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
    # Initialize RNN
    rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
    
    for epoch in range(100): 
        output = rnn.forward(input_sequence) 
        rnn.backward(input_sequence, expected_output, learning_rate=0.01) 
    print(output)