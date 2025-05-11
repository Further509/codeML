import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	"""
	# Your code here
	ans = 0
	for x, y in zip(vec1, vec2):
		ans += x * y
	return ans

if __name__ == "__main__":
	vec1 = np.array([1, 2, 3])
	vec2 = np.array([4, 5, 6])
	print(calculate_dot_product(vec1, vec2))