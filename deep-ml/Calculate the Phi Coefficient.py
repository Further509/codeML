import math

def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    # Your code here
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, j in zip(x, y):
        if i == 1 and j == 1: 
            tp += 1
        elif i == 1 and j == 0:
            tn += 1
        elif i == 0 and j == 1:
            fp += 1
        else:
            fn += 1
    val = (tp * fn - tn * fp) / math.sqrt((tp + tn) * (fp + fn) * (tp + fp) * (tn + fn)) 
    return round(val,4)