# %%
import re
from collections import Counter, defaultdict
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from ckiptagger import WS, POS
# %%
dat = pd.read_csv("../data/org_finews.csv", index_col=0).drop(
    columns=["Unnamed: 0"]
)
dat.dropna(inplace=True)
ws = WS("../model_data/data")
pos = POS("../model_data/data")
# %%
def clean(sents: List[str]) -> List[str]:
    sents = [
        s for s in sents if not (
            re.compile(r'[a-zA-Z]').search(s) or \
                re.compile(r'[0-9]').search(s) or \
                re.compile(r'[^\w\s]').search(s) or \
                s == '\u3000'
        )
    ]
    return sents


def combine(sents: List[List[str]], tags: List[List[str]]) -> List[List[tuple]]:
    output = []
    for sent in sents:
        output.append(
            [
                (word, tag) for word, tag in zip(
                    sent, tags[sents.index(sent)]
                )
            ]
        )
    return output


def get_spec_pos(lsts: List[List[tuple]]) -> List[List[str]]:
    output = []
    for lst in lsts:
        output.append(
            [
                pair[0] for pair in lst if (
                    pair[1][0] in ["N", "V", "A"]
                ) and (
                    len(pair[0]) > 1
                )
            ]
        )
    return output


def get_unique_words(lsts: List[List[str]]) -> List[str]:
    output = []
    for lst in lsts:
        output.extend(lst)
    return list(set(output))

# %%
class TfidfBuilder:
    def __init__(self, corpus: List[List[str]], words_set: List[str]) -> None:
        self.corpus = corpus
        self.words_set = words_set
        self.flatten_corpus = [word for doc in corpus for word in doc]
        self.n = len(corpus)

    def get_term_freq(self) -> Counter:
        return Counter(self.flatten_corpus)
        

    def get_doc_freq(self) -> Dict[str, int]:
        inverted_index = defaultdict(set)
        for docid, doc in enumerate(self.corpus):
            for word in doc:
                inverted_index[word].add(docid)
        doc_freq = {
            word: len(docids) for word, docids in inverted_index.items()
        }
        return doc_freq


    def cmb_tfidf(self) -> list:
        tf_dict, df_dict = self.get_term_freq(), self.get_doc_freq()
        return {
            word: tf * np.log(self.n / (df_dict[word] + 1)) for word, tf in tf_dict.items()
        }
# %%
segs = ws(
    dat["NewsContents"].tolist(),
    sentence_segmentation=True,
    segment_delimiter_set={'?', '？', '!', '！', '。', ',','，', ';', ':', '、'}
)
# %%
cleansegs = []
for idx, sent in tqdm(enumerate(segs)):
    cleansegs.append(clean(sent))

tagger = pos(cleansegs)
opt = combine(cleansegs, tagger)
opt2 = get_spec_pos(opt)
# %%
unq_words = get_unique_words(opt2)
# %%
tfidf_builder = TfidfBuilder(opt2, unq_words)
tfidf = tfidf_builder.cmb_tfidf()
