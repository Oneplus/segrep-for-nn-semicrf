#!/bin/bash
awk '{if(length($0)>0){print $1" "$3" "$4}else{print}}' $1 > $1.tmp
./scripts/tags_to_word.py $1.tmp
touch $1.dummy
./scripts/score $1.dummy $1.tmp.gold $1.tmp.pred | egrep '^=== F MEASURE:' | awk '{print $4}'
rm $1.dummy $1.tmp.gold $1.tmp.pred $1.tmp

