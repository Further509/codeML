import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	# Implement PMI calculation here
    joint = joint_counts / total_samples
    single = (total_counts_x / total_samples) * (total_counts_y / total_samples)
    res = np.log2(joint / single)
    return np.round(res, 3) 

if __name__ == "__main__":
    print(compute_pmi(50, 200, 300, 1000))
    print(compute_pmi(100, 400, 600, 1200))