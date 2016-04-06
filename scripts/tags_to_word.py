#!/usr/bin/env python
import sys
import os

dataset = open(sys.argv[1], "r").read().strip().split("\n\n")

fp_gold = open("%s.gold" % sys.argv[1], "w")
fp_pred = open("%s.pred" % sys.argv[1], "w")

for data in dataset:
    lines = data.split("\n")
    gold_words = []
    pred_words = []
    gold_word, pred_word = "", ""
    for line in lines:
        ch, gold, pred = line.split()
        if gold == "B" or gold == "S":
            if len(gold_word) > 0:
                gold_words.append(gold_word)
            gold_word = ch
        else:
            gold_word += ch

        if pred == "B" or pred == "S":
            if len(pred_word) > 0:
                pred_words.append(pred_word)
            pred_word = ch
        else:
            pred_word += ch
    if len(gold_word) > 0:
        gold_words.append(gold_word)
    if len(pred_word) > 0:
        pred_words.append(pred_word)

    print >> fp_gold, " ".join(gold_words)
    print >> fp_pred, " ".join(pred_words)
