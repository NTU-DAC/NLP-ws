from collection import Counter
import numpy as np
from tqdm import tqdm
from .tfidf import TFiDFBuilder


class NGram(TFiDFBuilder):
    def __init__(self, sentences: list, n: int) -> None:
        super(NGram, self).__init__(sentences)
        self.sentences = sentences
        self.num_sentences = len(sentences)


    def get_ngram(self, n: int=2) -> list:
        """
        get unigram from corpus
        """
        dic = dict()
        count = 0
		  for sentence in self.sentences:
            dic[count] = [sentence[i: i+n] for i in range(len(sentence)+n-1)]
            count += 1
        return lst
				
            

if __name__ == "__main__":
    n = 3
    builder = NGram(corpus)
    ngram_dict = builder.get_ngram(n)