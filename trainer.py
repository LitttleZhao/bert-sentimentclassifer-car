from logging.config import valid_ident
from multiprocessing import set_forkserver_preload
from utils.config import Config
from utils.dataset import SentimentDataset, convert_comment_to_ids, seq_padding

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,SequentialSampler,BatchSampler,SubsetRandomSampler

import numpy as np
import random
import transformers
from transformers import BertConfig,BertForSequenceClassification
from transformers import AdamW

# bert分类器类代码
class BertClassifier(object):
    def __init__(self,args):
        self.config = Config()
        self.model_set(args)
        
    def model_set(self,args):
        
        # 设置种子
        self.freezeSeed()
        # gpu 设置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        import os
        result_dir = args.result_dir
        if len(os.listdir(result_dir)) < 3:
            MODEL_PATH = self.config.get("BERT_path", "file_path")
            CONFIG_PATH = self.config.get("BERT_path", "config_path")
            VOCAB_PATH = self.config.get("BERT_path", "vocab_path")
        else:
            MODEL_PATH = self.config.get("result", "model_save_path")
            CONFIG_PATH = self.config.get("result", "config_save_path")
            VOCAB_PATH = self.config.get("result", "vocab_save_path")

        num_labels = self.config.get("training_rule", "num_labels")
        hidden_dropout_prob = self.config.get("training_rule", "hidden_dropout_prob")

        #分词器
        self.tokenizer = transformers.BertTokenizer.from_pretrained(VOCAB_PATH)
        self.model_config = BertConfig.from_pretrained(CONFIG_PATH,num_labels=num_labels,hidden_dropout_prob=hidden_dropout_prob)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH,config=self.model_config)

        self.model.to(self.device)

    def freezeSeed(self):
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def optimizer_and_loss_set(self):
        weight_decay = self.config.get("training_rule", "weight_decay")
        learning_rate = self.config.get("training_rule", "learning_rate")

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def data_set(self):

        train_set_path = self.config.get("data_path","trainSet_path")

        batch_size = self.config.get("training_rule", "batch_size")
        
        sentiment_train_set = SentimentDataset(train_set_path)
        
        validation_split = 0.2 # 选择20%作为测试数据
        dataset_size = len(sentiment_train_set)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split*dataset_size))
        np.random.shuffle(indices)  # 使用numpy来打乱数据
        train_indices,val_indices = indices[split:],indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)


        sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size,sampler=train_sampler, num_workers=5)
        sentiment_valid_loader = DataLoader(sentiment_train_set,batch_size=batch_size,sampler=valid_sampler,num_workers=5)

        return sentiment_train_loader,sentiment_valid_loader

    def train_epoch(self,args,iterator):
        self.optimizer_and_loss_set()
        epoch_loss = 0
        epoch_acc = 0

        for i,batch in enumerate(iterator):
            label = batch["label"]
            comment = batch["comment"]

            input_ids,token_type_ids = convert_comment_to_ids(self.tokenizer,comment)
            input_ids = seq_padding(self.tokenizer,input_ids)
            token_type_ids = seq_padding(self.tokenizer,token_type_ids)
            label = [int(i) for i in label]
            label = torch.tensor(label)
            label = label.unsqueeze(1)

            input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
            # input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            self.optimizer.zero_grad() # 梯度清零

            input_ids,token_type_ids,label = input_ids.to(self.device),token_type_ids.to(self.device),label.to(self.device)

            output = self.model(input_ids=input_ids,token_type_ids=token_type_ids,labels=label)

            logits = output[1]

            logits_label = logits.argmax(dim=1)

            loss1 = output[0]
            loss = self.criterion(logits.view(-1,2),label.view(-1))

            acc = ((logits_label == label.view(-1)).sum()).item()

            # 反向传播
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            if i % 20 == 0:
                print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", epoch_acc / ((i + 1) * len(label)))
        
        cur_loss = epoch_loss / len(iterator)
        cur_acc = epoch_acc / ( len(iterator.dataset.dataset)*0.8)
        
        return cur_loss,cur_acc 


    def evaluate(self,args,iterator,test=False,cur_max_acc=0):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for _,batch in enumerate(iterator):
                label = batch["label"]
                comment = batch["comment"]

                input_ids,token_type_ids = convert_comment_to_ids(self.tokenizer,comment)
                input_ids = seq_padding(self.tokenizer,input_ids)
                token_type_ids = seq_padding(self.tokenizer,token_type_ids)
                label = label.unsqueeze(1)
                input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
                input_ids,token_type_ids,label = input_ids.to(self.device),token_type_ids.to(self.device),label.to(self.device)
                output = self.model(input_ids=input_ids,token_type_ids=token_type_ids,labels=label)
                logits = output[1]
                logits_label = logits.argmax(dim=1)
                loss = output[0]
                # loss = self.criterion(logits.view(-1,2),label.view(-1,2))

                acc = ((logits_label == label.view(-1)).sum()).item()

                epoch_loss += loss.item()
                epoch_acc += acc

        cur_loss = epoch_loss / len(iterator)
        cur_acc = epoch_acc / ( len(iterator.dataset.dataset) * 0.2 )
        self.save_model(args)
        return cur_loss, cur_acc
        

    def train(self,args):
        train_loader,valid_loader = self.data_set()
        cur_max_acc = 0

        for i in range(1,args.epoch+1):
            train_loss, train_acc = self.train_epoch(args,train_loader)
            print("epoch:{}, train loss:{:.3f}, train acc:{:.3f}".format(i,train_loss,train_acc))

            valid_loss, valid_acc = self.evaluate(args,valid_loader,test = False)
            print("epoch:{}, valid loss:{:.3f}, valid acc:{:.3f}".format(i,valid_loss,valid_acc))

            # test_loss, test_acc ,cur_max_acc = self.evaluate(args, test_loader,test = True,cur_max_acc = cur_max_acc)
            # print("epoch:{}, test loss:{:.3f}, test acc:{:.3f}".format(i,test_loss,test_acc))
    
    def save_model(self, args=None):
        model_save_path = self.config.get("result", "model_save_path") + args.my_model+"_model.bin"
        config_save_path = self.config.get("result", "config_save_path") + args.my_model+"_config.json"
        vocab_save_path = self.config.get("result", "vocab_save_path") + args.my_model+"_vocab.txt"

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), model_save_path)
        model_to_save.config.to_json_file(config_save_path) # !!!'bert_lr' object has no attribute 'config'
        self.tokenizer.save_vocabulary(vocab_save_path)
        print("model saved...")
    
    def predict(self,sentence):
        self.optimizer_and_loss_set()
        self.model.eval()

        input_ids, token_type_ids = convert_comment_to_ids(self.tokenizer, sentence)
        input_ids = seq_padding(self.tokenizer, [input_ids])
        token_type_ids = seq_padding(self.tokenizer, [token_type_ids])

        input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
        self.optimizer.zero_grad()
        input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = output[0]
        logits_label = logits.argmax(dim=1)

        return logits_label.item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",default="chinese-bert-wwm",type=str)
    parser.add_argument("--my_model",default="huggingface_default",type=str)
    parser.add_argument("--epoch",default="8",type=int)
    parser.add_argument("--result_dir",default="/root/data/user/bert/result",type=str)
    parser.add_argument("--do_train", default= True,action="store_true",help="Whether to run training.")
    parser.add_argument("--do_predict",default= True,action="store_true",help="Whether to run predictions on the test set.")
    args = parser.parse_args()


    classifier = BertClassifier(args)
    if args.do_train:
        classifier.train(args)

    if args.do_predict:
        print(classifier.predict("空间没的说，排坐的都很舒适，尤其第二排，大家都抢着要做"))  # 1
        print(classifier.predict("中间乘坐空间无敌，第排也够用，唯一缺点就是后备箱空间小了点，但是日常城市使用足够，旅游的话6个人行李会很紧张"))  # 0
        print(classifier.predict("后排空间没得说，第三排而且我是长年放倒基本用不到"))  # 1
        print(classifier.predict("这个车是七座的，但是车身的长度只有4米8。和一个中型的SUV差不多。能塞下七个座椅真的是挺费劲的，第三排的位置并不是很大"))  # 0
        print(classifier.predict("空间巨大，大四座能坐1.8米大长腿美女。折叠三非后，装载异常大，地台低也好拿放。"))  # 1
        print(classifier.predict("5分吧，，，其实就是尾箱。。。如果再大一点。去到5米车长，就非常理想咯"))  # 0
