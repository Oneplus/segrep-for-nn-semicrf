#!/bin/bash
root=`dirname $BASH_SOURCE`
./$root/conlleval < $1 | egrep '^accuracy:' | awk -F "FB1:" '{print $2}' | sed 's/^[ \t]*//g'
