#!/bin/bash

./bin/labeler -m $1 \
    --graph bilstm \
    -T ./data_sample/cws/cws.train.tag \
    -d ./data_sample/cws/cws.test.tag \
    -w ./data_sample/cws/giga.cbow.100.sample \
    --pos_dim 0 \
    --layers 2 \
    --pretrained_dim 100 \
    --lstm_input_dim 100 \
    --hidden_dim 100 \
    --conlleval ./scripts/cws_label_eval.sh
