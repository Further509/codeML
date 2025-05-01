import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    if not reference or not candidate:
        raise ValueError("Reference and candidate cannot be empty")
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    ref_counts = Counter(ref_tokens) 
    cand_counts = Counter(cand_tokens)

    num_matches = sum((ref_counts & cand_counts).values())
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)  

    precision = num_matches / cand_len if cand_len > 0 else 0 
    recall = num_matches / ref_len if ref_len > 0 else 0 
    
    if num_matches == 0:
        return 0.0
    
    fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    # Chunk calculation 
    matched_positions = []
    ref_positions = {}  # Store positions of words in reference
    used_positions = set()  # Track already used indices

    # Populate reference positions for word alignment tracking
    for i, word in enumerate(ref_tokens):
        ref_positions.setdefault(word, []).append(i)

    # Determine the sequence of matched positions in reference
    for word in cand_tokens:
        if word in ref_positions:
            for pos in ref_positions[word]:
                if pos not in used_positions:
                    matched_positions.append(pos)
                    used_positions.add(pos)
                    break  # Ensure each match is used only once

    num_chunks = 1 if matched_positions else 0
    for i in range(1, len(matched_positions)):
        if matched_positions[i] != matched_positions[i - 1] + 1:
            num_chunks += 1

    penalty = gamma * ((num_chunks / num_matches) ** beta) if num_matches > 0 else 0

    return round(fmean * (1 - penalty), 3) 


if __name__ == "__main__":
    ans = meteor_score('Rain falls gently from the sky', 'Gentle rain drops from the sky')
    print(ans)