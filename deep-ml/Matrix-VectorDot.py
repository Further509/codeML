def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	a_row, a_col = len(a), len(a[0])
	b_row = len(b)
	if a_col != b_row:
		return -1
	ans = []
	for a_i in a:
		total = 0
		for j in range(b_row):
			total += a_i[j] * b[j]
		ans.append(total)

	return ans

if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    b = [7, 8, 9]
    print(matrix_dot_vector(a, b)) # [50, 122]
    a = [[1, 2], [3, 4]]
    b = [5, 6]
    print(matrix_dot_vector(a, b)) # [17, 39]
    a = [[1, 2], [3, 4]]
    b = [5]
    print(matrix_dot_vector(a, b)) # -1