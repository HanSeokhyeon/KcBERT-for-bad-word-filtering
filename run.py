# -*- coding: utf-8 -*-
# import time
from config.arg_badword_pretrained import Arg
from model_for_inference import Model

args = Arg()
# 모델 통과
print("Model importing...")
model = Model(args)


def inference(sentence):
    # start = time.time()
    logits, pred = model.inference(sentence)
    # print("processing time {}s".format(time.time() - start))
    return logits, pred


if __name__ == '__main__':
    probability, label = inference("돈이 있다는 조건으로. 돈없는 대학 생활은 좃이제.")
    print(probability, label)
