import numpy as np

def calculate_contrast(img) -> int:
    """
    Calculate the contrast of a grayscale image.
    Args:
        img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
    """
    # Your code here
    m, n = len(img), len(img[0])
    mini, maxm = 255, 0
    for i in range(m):
        for j in range(n):
            mini = min(mini, img[i][j])
            maxm = max(maxm, img[i][j])
    return maxm - mini

if __name__ == "__main__":
    img = np.array([[0, 50], [200, 255]])
    print(calculate_contrast(img))
    print(calculate_contrast(np.array([[128, 128], [128, 128]])))
    print(calculate_contrast(np.zeros((10, 10), dtype=np.uint8)))
    print(calculate_contrast(np.ones((10, 10), dtype=np.uint8) * 255))
    print(calculate_contrast(np.array([[10, 20, 30], [40, 50, 60]])))