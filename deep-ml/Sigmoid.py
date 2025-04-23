import math

def sigmoid(z: float) -> float:
	#Your code here
    ans = 1 / (1 + math.exp(-z))
    return round(ans, 4)