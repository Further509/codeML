import numpy as np


def compute_tf_idf(corpus, query):
    if not corpus:
        return []

    # 计算每个词在每个文档中的词频（TF）
    def compute_tf(doc, word):
        return doc.count(word) / len(doc) if doc else 0

    # 计算每个词的逆文档频率（IDF）
    def compute_idf(word):
        doc_count = sum([1 for doc in corpus if word in doc])
        # 处理df为0的情况，避免除零错误
        return np.log((len(corpus) + 1) / (doc_count + 1)) + 1

    result = []
    for doc in corpus:
        doc_scores = []
        for q_word in query:
            tf = compute_tf(doc, q_word)
            idf = compute_idf(q_word)
            # print(tf, idf)
            tf_idf = tf * idf
            doc_scores.append(round(tf_idf, 5))
        result.append(doc_scores)

    return result

if __name__ == "__main__":
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["the", "bird", "flew", "over", "the", "mat"]
    ]
    query = ["cat", "the"]

    print(compute_tf_idf(corpus, query))