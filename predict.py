from trainer import BertClassifier
import torch
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir",default="/root/data/user/bert/result",type=str)
parser.add_argument("--do_predict",default= True,action="store_true",help="Whether to run predictions on the test set.")
args = parser.parse_args()
classifier = BertClassifier(args)

df = pd.read_csv("/root/data/user/bert/data/data_pred(1).csv",header=None)
for i in range(len(df)):
    print(classifier.predict(df.loc[i,0]))

# print(classifier.predict("空间绝对够用，只是车子没后备箱，这个让我有些沮丧。"))  # 0
# print(classifier.predict("中间乘坐空间无敌，第排也够用，唯一缺点就是后备箱空间小了点，但是日常城市使用足够，旅游的话6个人行李会很紧张"))  # 0
# print(classifier.predict("后排空间没得说，第三排而且我是长年放倒基本用不到"))  # 1
# print(classifier.predict("这个车是七座的，但是车身的长度只有4米8。和一个中型的SUV差不多。能塞下七个座椅真的是挺费劲的，第三排的位置并不是很大"))  # 0
# print(classifier.predict("空间巨大，大四座能坐1.8米大长腿美女。折叠三非后，装载异常大，地台低也好拿放。"))  # 1
# print(classifier.predict("5分吧，，，其实就是尾箱。。。如果再大一点。去到5米车长，就非常理想咯"))  # 0