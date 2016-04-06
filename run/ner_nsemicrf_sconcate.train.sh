#!/bin/bash

./bin/semi_crf -t \
    --graph concate \
    -T ./data_sample/ner/ner.train.seg \
    -d ./data_sample/ner/ner.devel.seg \
    -w ./data_sample/ner/sskip.100.vector.sample \
    --unk_strategy 1 \
    --unk_prob 0.2 \
    --pretrained_dim 100 \
    --lstm_input_dim 100 \
    --hidden1_dim 100 \
    --hidden2_dim 100 \
    --conlleval ./scripts/ner_eval.sh \
    --maxiter 30
