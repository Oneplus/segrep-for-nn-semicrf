#!/bin/bash

./bin/crf -t \
    -T ./data_sample/cws/cws.train.tag \
    -d ./data_sample/cws/cws.devel.tag \
    -w ./data_sample/cws/giga.cbow.100.sample \
    --unk_strategy 0 \
    --pos_dim 0 \
    --pretrained_dim 100 \
    --layers 2 \
    --hidden_dim 100 \
    --lstm_input_dim 100 \
    --conlleval ./scripts/cws_label_eval.sh \
    --maxiter 30
