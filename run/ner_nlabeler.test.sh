#!/bin/bash

./bin/labeler -m $1 \
    --graph bilstm \
    -T ./data_sample/ner/ner.train.tag \
    -d ./data_sample/ner/ner.test.tag \
    -w ./data_sample/ner/sskip.100.vector.sample \
    --pretrained_dim 100 \
    --hidden_dim 100 \
    --lstm_input_dim 100 \
    --conlleval ./scripts/ner_eval.sh
