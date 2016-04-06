#!/bin/bash
./scripts/tags_to_word.py $1
touch $1.dummy
./scripts/score $1.dummy $1.gold $1.pred | egrep '^=== F MEASURE:' | awk '{print $4}'
rm $1.dummy $1.gold $1.pred

