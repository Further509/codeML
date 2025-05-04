import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    # Your code here
    input_sequence = np.array(input_sequence)
    hidden_state = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    for input_vector in input_sequence:
        hidden_state = np.tanh(np.dot(Wx, input_vector) + np.dot(Wh, hidden_state) + b)

    final_hidden_state = np.round(hidden_state, 4).tolist()
    return final_hidden_state

if __name__ == "__main__":
    input_sequence = [[1.0], [2.0], [3.0]]
    initial_hidden_state = [0.0]
    Wx = [[0.5]]  # Input to hidden weights
    Wh = [[0.8]]  # Hidden to hidden weights
    b = [0.0]     # Bias
    ans = rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b)
    print(ans)