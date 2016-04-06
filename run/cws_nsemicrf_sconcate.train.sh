#!/bin/bash

./bin/semi_crf2 -t \
    --graph zero_concate \
    -T ./data_sample/cws/cws.train.seg \
    -d ./data_sample/cws/cws.devel.seg \
    -w ./data_sample/cws/giga.cbow.100.sample \
    --conlleval ./scripts/cws_semicrf_eval.sh \
    --unk_strategy 0 \
    --layers 2 \
    --lstm_input_dim 100 \
    --hidden1_dim 100 \
    --hidden2_dim 100 \
    --maxiter 30
