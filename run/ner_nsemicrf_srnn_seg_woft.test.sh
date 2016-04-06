#!/bin/bash

./bin/semi_crf -m $1 \
    --graph rnn_rich_lex \
    -T ./data_sample/ner/ner.train.seg \
    -d ./data_sample/ner/ner.test.seg \
    -w ./data_sample/ner/sskip.100.vector.sample \
    --lexicon ./data_sample/ner/ner.segemb.sample \
    --pretrained_dim 100 \
    --lstm_input_dim 100 \
    --hidden1_dim 100 \
    --hidden2_dim 100 \
    --conlleval ./scripts/ner_eval.sh
