from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm

class TfidfBuilder:
    """
    tfidf builder
    """
    def __init__(self, corpus: List[List[str]], words_set: List[str]) -> None:
        self.corpus = corpus
        self.words_set = words_set
        self.flatten_corpus = [word for doc in corpus for word in doc]
        self.n = len(corpus)

    def get_term_freq(self) -> Counter:
        """
        get term frequency
        """
        return Counter(self.flatten_corpus)


    def get_doc_freq(self) -> Dict[str, int]:
        """
        get document frequency
        """
        inverted_index = defaultdict(set)
        for docid, doc in tqdm(enumerate(self.corpus)):
            for word in doc:
                inverted_index[word].add(docid)
        doc_freq = {
            word: len(docids) for word, docids in inverted_index.items()
        }
        return doc_freq


    def get_tfidf(self) -> list:
        """
        get tfidf
        """
        tf_dict, df_dict = self.get_term_freq(), self.get_doc_freq()
        return {
            word: tf * np.log(self.n / (df_dict[word] + 1)) for word, tf in tf_dict.items()
        }