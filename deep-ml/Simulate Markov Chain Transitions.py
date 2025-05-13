import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    state = [initial_state]
    current_state = initial_state
    for _ in range(num_steps):
        probabilities = transition_matrix[current_state]
        next_state = np.random.choice(len(probabilities), p=probabilities)
        state.append(next_state)
        current_state = next_state
    return np.array(state)

if __name__ == "__main__":
    np.random.seed(42)
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    print(simulate_markov_chain(transition_matrix, 0, 3))