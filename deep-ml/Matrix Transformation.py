import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A = np.array(A, dtype=float)
	T = np.array(T, dtype=float)
	S = np.array(S, dtype=float)
	if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
		return -1
	inv_T = np.linalg.inv(T)
	ans = inv_T @ A @ S
	return ans.tolist()

if __name__ == "__main__":
	A = [[1, 2], [3, 4]]
	T = [[2, 0], [0, 2]] 
	S = [[1, 1], [0, 1]] 
	output = transform_matrix(A, T, S)
	print(output)