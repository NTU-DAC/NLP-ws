"""
Tfidf script
"""
from collections import Counter
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

class TFiDFBuilder:
    """
    Building tf-idf vector
    """
    def __init__(self, sentences: list) ->  None:
        self.sentences = sentences
        self.num_sentences = len(sentences)

    def get_unique_words(self) -> list:
        """
        get unique words from corpus
        """
        unique_words = list(set(word_tokenize(" ".join(self.sentences))))
        return unique_words


    def get_tf(self, sentence: str) -> Counter:
        """
        Get term-frequency
        """
        tf = Counter(word_tokenize(sentence))
        return tf


    def get_df(self, unique_words: list) -> Counter:
        """
        get document-frequency
        """
        df = Counter()
        for word in unique_words:
            for sentence in self.sentences:
                if word in word_tokenize(sentence):
                    df[word] += 1
        return df


    def get_tfidf(self) -> dict:
        """
        get tf-idf vector based on given tf and idf
        """
        unique_words = self.get_unique_words()
        tfidf = {}
        df = self.get_df(unique_words)
        for sentence in tqdm(self.sentences):
            tf = self.get_tf(sentence)
            for word in tf:
                tfidf[word] = tf.get(word, 0) * np.log(self.num_sentences / df[word])
        return tfidf


if __name__ == "__main__":
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        corpus = f.readlines()
    builder = TFiDFBuilder(corpus)
    print(builder.get_tfidf())
