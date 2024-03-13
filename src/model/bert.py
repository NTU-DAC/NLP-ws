import torch
import torch.nn as nn
import transformers


class MyBert(nn.Module):
    def __init__(self, model_name, num_classes) -> None:
        super(MyBert, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)