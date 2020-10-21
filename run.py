# -*- coding: utf-8 -*-
import time
import pandas as pd

from config.arg_badword_pretrained_puri_2400 import Arg
from model_for_inference import Model

args = Arg()
# 모델 통과
print("Model importing...")
model = Model(args)
model.eval()


def inference(sentence):
    start = time.time()
    logits, pred, sentence_filtered = model.inference(sentence)
    print("processing time {:.4f}s".format(time.time() - start))
    return logits, pred


if __name__ == '__main__':
    test = pd.read_csv("badword/ratings_labeled_test.csv")
    for i in range(100):
        text, label = test.iloc[i]['document'], test.iloc[i]['label']
        probability, pred = inference(text)
        print("label: {},\tpred: {},\tprobability: {:.2f}\t{:.2f}\n".format(label, pred, *probability))
