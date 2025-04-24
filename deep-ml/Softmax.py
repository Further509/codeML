import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    probabilities = scores.copy()
    for i, score in enumerate(scores):
        probabilities[i] = math.exp(score)
    total = sum(probabilities)
    for i, score in enumerate(probabilities):
        probabilities[i] = round(score / total, 4)
    return probabilities

if __name__ == "__main__":
    scores = [1.0, 2.0, 3.0]
    probabilities = softmax(scores)
    print("Softmax Probabilities:", probabilities)