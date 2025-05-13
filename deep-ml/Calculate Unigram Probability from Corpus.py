def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    corpus_list = corpus.split(' ')
    return round(corpus_list.count(word) / len(corpus_list), 4)

if __name__ == "__main__":
    corpus = "<s> Jack I like </s> <s> Jack I do like </s>"
    word = "Jack"
    print(unigram_probability(corpus, word))