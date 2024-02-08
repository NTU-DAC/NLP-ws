import re
from typing import List
from tqdm import tqdm


class TextPreproc:
    """
    ZH text preprocessor
    """
    def __init__(self, pos, tokens: List[List[str]]) -> None:
        self.pos = pos
        self.tokens = tokens


    def clean(self, sents: List[str]) -> List[str]:
        """
        cleaning words
        """
        sents = [
            s for s in sents if not (
                re.compile(r'[a-zA-Z]').search(s) or \
                    re.compile(r'[0-9]').search(s) or \
                    re.compile(r'[^\w\s]').search(s) or \
                    s == '\u3000'
            )
        ]
        return sents


    def combine(
            self, sents: List[List[str]], tags: List[List[str]]
        ) -> List[List[tuple]]:
        """
        combine words and pos_tags
        """
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


    def get_spec_pos(self, lsts: List[List[tuple]]) -> List[List[str]]:
        """
        get specific pos_tags words
        """
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


    def main(self) -> List[List[str]]:
        """
        main function
        """
        cleansegs = []
        for _, sent in tqdm(enumerate(self.tokens)):
            cleansegs.append(self.clean(sent))
        tags = self.pos(cleansegs)
        combined = self.combine(cleansegs, tags)
        spec_pos_words = self.get_spec_pos(combined)
        return spec_pos_words
