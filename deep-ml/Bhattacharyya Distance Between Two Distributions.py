import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    # Your code here
    if len(p) != len(q) or len(p) == 0: return 0.0
    bc = 0
    for p1, q1 in zip(p, q):
        bc += np.sqrt(p1 * q1)
    return round(-np.log(bc), 4) if bc > 0 else 0.0

if __name__ == "__main__":
    p = [0.1, 0.2, 0.3, 0.4] 
    q = [0.4, 0.3, 0.2, 0.1]
    print(bhattacharyya_distance(p, q))