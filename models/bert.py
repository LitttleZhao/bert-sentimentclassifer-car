import logging
import os
import torch.nn as nn
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset
from tqdm import tqdm,trange

from transformers import BertConfig,BertForSequenceClassification

# bert-config
class BertConfig(nn.Model):
    def __init__(self):
        self.bert_path = "../_pre_trained_model/chinese-bert-wwm"
        self.config_path = "../_pre_trained_model/chinese-bert-wwm/config.json"

        self.num_label = 2
        self.dropout_bertout = 0.2
        self.trained_model = "./result/bert_clf_model.bin"


