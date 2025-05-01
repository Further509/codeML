
def calculate_brightness(img):
    # Write your code here
    if not img:
        return -1
    total = 0
    m = len(img)
    n = len(img[0])
    for i in range(m):
        if len(img[i]) != n:
            return -1
        for j in range(n):
            if img[i][j] < 0 or img[i][j] > 255:
                return -1
            total += img[i][j]
    return round(total / (m * n), 2)
