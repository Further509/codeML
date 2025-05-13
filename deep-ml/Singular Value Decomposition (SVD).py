import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    a  = A
    a_t = np.transpose(a)
    a_2 = a_t @ a 
    v = np.eye(2)

    if a_2[0, 0] == a_2[1, 1]:
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan2(2 * a_2[0, 1], a_2[0, 0] - a_2[1, 1])

    r = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]
    )  

    d = np.transpose(r) @ a_2 @ r

    a_2 = d

    v = v @ r

    s = np.sqrt([d[0, 0], d[1, 1]])
    s_inv = np.array([[1/s[0], 0], [0, 1/s[1]]])

    u = a @ v @ s_inv

    return (u, s, v.T)
 
if __name__ == "__main__":
    A = np.array([[2, 1], [1, 2]])
    print(svd_2x2_singular_values(A))