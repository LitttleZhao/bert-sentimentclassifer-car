from cProfile import label
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader,SequentialSampler
import numpy as np
# 读取样本
class SentimentDataset(Dataset):
    def __init__(self,file_path):
        self.dataset = pd.read_csv(file_path,sep=',',names=['comment','label'])
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self,idx):
        comment = self.dataset.loc[idx,"comment"]
        label = self.dataset.loc[idx,"label"]
        sample = {"comment":comment,"label":label}
        return sample

def convert_comment_to_ids(tokenizer, comment, max_len=100):
    if isinstance(comment, str):
        tokenized_comment = tokenizer.encode_plus(comment, max_length=max_len, add_special_tokens=True, truncation=True)
        input_ids = tokenized_comment["input_ids"]
        token_type_ids = tokenized_comment["token_type_ids"]
    elif isinstance(comment, list):
        input_ids = []
        token_type_ids = []
        for t in comment:
            tokenized_comment = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True, truncation=True)
            input_ids.append(tokenized_comment["input_ids"])
            token_type_ids.append(tokenized_comment["token_type_ids"])
    else:
        print("Unexpected input")
    return input_ids, token_type_ids


def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X


train_set_path = "/root/data/user/bert/data/data.csv"
        
sentiment_train_set = SentimentDataset(train_set_path)
dataset_size = len(sentiment_train_set)
indices = list(range(dataset_size))
np.random.shuffle(indices)  # 使用numpy来打乱数据
train_indices = indices
train_sampler = SequentialSampler(train_indices)
sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=64,sampler=train_sampler, num_workers=3)

