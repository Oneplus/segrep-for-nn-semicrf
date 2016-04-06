#!/bin/bash

./bin/semi_crf2 -m $1 \
    --graph zero_rnn \
    -T ./data_sample/cws/cws.train.seg \
    -d ./data_sample/cws/cws.test.seg \
    -w ./data_sample/cws/giga.cbow.100.sample \
    --conlleval ./scripts/cws_semicrf_eval.sh \
    --layers 2 \
    --lstm_input_dim 100 \
    --hidden1_dim 100 \
    --hidden2_dim 100
