import torch.nn as nn
# 路径及属性配置
class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                "trainSet_path": "/root/data/user/bert/data/data.csv",
                "testSet_path": "/root/data/user/bert/data/data_pred(1).csv"
            },

            "BERT_path": {
                "file_path": '/root/data/user/bert/chinese-bert-wwm/',
                "config_path": '/root/data/user/bert/chinese-bert-wwm/',
                "vocab_path": '/root/data/user/bert/chinese-bert-wwm/',
            },

            "training_rule": {
                "max_length": 300,  
                "hidden_dropout_prob": 0.3,
                "num_labels": 2, 
                "learning_rate": 1e-5,
                "weight_decay": 1e-2,
                "batch_size": 64
            },

            "result": {
                "model_save_path": '/root/data/user/bert/result/',
                "config_save_path": '/root/data/user/bert/result/',
                "vocab_save_path": '/root/data/user/bert/result/'
            },
            "result1": {
                "model_save_path": '/root/data/user/bert/result/huggingface_default_model.bin',
                "config_save_path": '/root/data/user/bert/result/huggingface_default_config.json',
                "vocab_save_path": '/root/data/user/bert/result/huggingface_default_vocab.txt'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]

