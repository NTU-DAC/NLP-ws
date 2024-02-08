import warnings
import pandas as pd
from ckiptagger import WS, POS
import src.utils.text_utils as tu
from src.preproc.text_preproc import TextPreproc
from src.tfidf import TfidfBuilder
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    dat = pd.read_csv("data/org_finews.csv", index_col=0).drop(
        columns=["Unnamed: 0"]
    )
    dat.dropna(inplace=True)
    tokenizer, tagger = WS("model_data/data"), POS("model_data/data")
    tokens = tu.tokenize(tokenizer, dat)
    text_preproc = TextPreproc(tagger, tokens)
    cleaned_tokens = text_preproc.main()
    unique_words = tu.get_unique_words(tokens)
    tfidf_builder = TfidfBuilder(cleaned_tokens, unique_words)
    tfidf = tfidf_builder.get_tfidf()
    print(tfidf) ## TODO: sort values
    ## TODO: build LM
    