import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """
        seq_length = x.shape[0]
        hidden_state = []
        h = initial_hidden_state
        c = initial_cell_state

        for t in range(seq_length):
            xt = x[t].reshape(-1, 1)
            combined = np.vstack((h, xt)) 

            ft = sigmoid(np.dot(self.Wf, combined) + self.bf)

            it = sigmoid(np.dot(self.Wi, combined) + self.bi)

            c_tilde = tanh(np.dot(self.Wc, combined) + self.bc)

            c = ft * c + it * c_tilde

            out = sigmoid(np.dot(self.Wo, combined) + self.bo)

            h = out * tanh(c)

            hidden_state.append(h.T)

        final_h = h.T
        final_c = c.T
        hidden_state = np.array(hidden_state)

        return hidden_state, final_h, final_c

if __name__ == "__main__":
    input_sequence = np.array([[1.0], [2.0], [3.0]])
    initial_hidden_state = np.zeros((1, 1))
    initial_cell_state = np.zeros((1, 1))

    lstm = LSTM(input_size=1, hidden_size=1)
    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

    print(final_h)

    input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]]) 
    initial_hidden_state = np.zeros((2, 1)) 
    initial_cell_state = np.zeros((2, 1)) 
    lstm = LSTM(input_size=2, hidden_size=2) # Set weights and biases for reproducibility 
    lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
    lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
    lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
    lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
    lstm.bf = np.array([[0.1], [0.2]]) 
    lstm.bi = np.array([[0.1], [0.2]]) 
    lstm.bc = np.array([[0.1], [0.2]]) 
    lstm.bo = np.array([[0.1], [0.2]]) 
    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state) 
    print(final_h)