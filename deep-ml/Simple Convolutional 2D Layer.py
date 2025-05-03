import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Your code here
    padding_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    output_height = int((input_height + 2 * padding - kernel_height) / stride) + 1
    output_width = int((input_width + 2 * padding - kernel_width) / stride) + 1

    output_matrix = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            start_row = i * stride
            end_row = start_row + kernel_height
            start_col = j * stride
            end_col = start_col + kernel_width
            conv_region = padding_matrix[start_row: end_row, start_col: end_col]
            output_matrix[i][j] = np.sum(conv_region * kernel)

    return output_matrix

if __name__ == "__main__":
    input_matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    kernel = np.array([
        [1, 0],
        [-1, 1]
    ])

    padding = 1
    stride = 2

    output = simple_conv2d(input_matrix, kernel, padding, stride)
    print(output)