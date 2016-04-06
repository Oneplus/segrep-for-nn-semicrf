#!/bin/bash

./bin/labeler -t \
    --graph bilstm \
    -T ./data_sample/ner/ner.train.tag \
    -d ./data_sample/ner/ner.devel.tag \
    -w ./data_sample/ner/sskip.100.vector.sample \
    --unk_strategy 1 \
    --unk_prob 0.2 \
    --pretrained_dim 100 \
    --hidden_dim 100 \
    --lstm_input_dim 100 \
    --conlleval ./scripts/ner_eval.sh \
    --maxiter 30
