import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    # Your code here
    num_docs = len(corpus)
    avg_doc_len = np.mean([len(doc) for doc in corpus])
    
    # 计算每个词的逆文档频率（IDF）
    def compute_idf(word):
        doc_count = sum([1 for doc in corpus if word in doc])
        # 处理df为0的情况，避免除零错误
        return np.log((len(corpus) + 1) / (doc_count + 1)) 
    
    scores = []
    for doc in corpus:
        doc_score = 0
        doc_len = len(doc)
        term_counts = Counter(doc)
        for term in query:
            tf = term_counts[term] 
            idf = compute_idf(term)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            if denominator != 0:
                term_score = idf * (numerator / denominator)
                doc_score += term_score
        scores.append(doc_score)

    return np.round(scores,3)

if __name__ == "__main__":
    corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']]
    query = ['the', 'cat']
    ans  = calculate_bm25_scores(corpus, query)
    print(ans)