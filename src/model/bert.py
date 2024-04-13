from transformers import BertModel, BertTokenizer


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    TEXT = "Hello, my dog is cute"
    embedded_word = tokenizer(TEXT, return_tensors="pt")
    