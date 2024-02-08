from typing import List
import pandas as pd


def tokenize(ws, data: pd.DataFrame):
    """
    ckip tokenizer
    """
    segs = ws(
        data["NewsContents"].tolist(),
        sentence_segmentation=True,
        segment_delimiter_set={
            '?', '？', '!', '！', '。', ',','，', ';', ':', '、'
        }
    )
    return segs


def get_unique_words(lsts: List[List[str]]) -> List[str]:
    """
    get unique words from tokens
    """
    output = []
    for lst in lsts:
        output.extend(lst)
    return list(set(output))
