Code for Exploring Segment Representations for Neural Segmentation Models
=========================================================================

This is the code base for our IJCAI 2016 paper.

# Prerequisite

1. cmake (~2.8)
2. git (~1.8)
3. g++ (~4.6 for c++11 features, 4.8 is used in this paper)
4. boost (~1.57)

# Compile

Execute the following command to compile.

```
git submodule init
git submodule update
./configure
make
```

You should find the following executable files:

* `./bin/crf`: the neural CRF baseline
* `./bin/labeler`: the neural classifier baselin
* `./bin/semi_crf`: the neural semi-CRF with prediction on segment label (for Named entity recognition)
* `./bin/semi_crf2`: the neural semi-CRF without prediction on segment label (for Chinese word segmentation) 

# Data format

## Input format

### .tag file

Used in `./bin/labeler` and `./bin/crf`.
Same with CoNLL03 format. Instances are separated by empty line. Each word in one instance occupy one line.
See `./data_sample/ner/ner.train.tag` for NER example and `./data_sample/cws/cws.train.tag` for CWS example.

### .seg file

Used in `./bin/semi_crf` and `./bin/semi_crf2`.
Each instance in one line with `|||` separating words and segmentation.
See `./data_sample/ner/ner.train.seg` for NER example and `./data_sample/cws/cws.train.seg` for CWS example.

### input unit embedding file

In the same format with word2vec.

### segment embedding file

Similar to the word2vec embedding format, but entry and its vector are separated by tab.
Since each entry (segment) consists one or more input units.
Surface strings of its units are separated by space.
See `./data_sample/ner/ner.segemb.sample` for named entity embedding example and `./data_sample/cws/cws.segemb.sample for Chinese word example.

# Running

Replace the `./data_sample/ner/ner.{train|devel|test}.tag` with CoNLL03 data to reproduce the NER result in the paper.

## Baseline

### NN-Labeler

Taking ner for example, execute to train a model on sample data.

```
./run/ner_nlabeler.train.sh
```

look for the model under root dir with name of `ner_bilstm_${args}.${pid}.params` and execute

```
./run/ner_nlabeler.test.sh ner_bilstm_${args}.${pid}.params
```

to perform test process.

### NN-CRF

* `./run/ner_ncrf.train.sh`
* `./run/ner_ncrf.test.sh crf_${args}.${pid}.params`

## Neural Semi-CRF

### SRNN

* `./run/ner_nsemicrf_srnn.train.sh`
* `./run/ner_nsemicrf_srnn.test.sh semi_crf_${args}.${pid}.params`

### SCONCATE

* `./run/ner_nsemicrf_sconcate.train.sh`
* `./run/ner_nsemicrf_sconcate.test.sh semi_crf_${args}.${pid}.params`

### SRNN+seg-embed

**With Fine Tuning**

* `./run/ner_nsemicrf_srnn_seg_wft.train.sh`
* `./run/ner_nsemicrf_srnn_seg_wft.test.sh semi_crf_${args}.${pid}.params`

**Without Fine Tuning**

* `./run/ner_nsemicrf_srnn_seg_woft.train.sh`
* `./run/ner_nsemicrf_srnn_seg_woft.test.sh semi_crf_${args}.${pid}.params`

### SCONCATE+seg-embed

**With Fine Tuning**

* `./run/ner_nsemicrf_sconcate_seg_wft.train.sh`
* `./run/ner_nsemicrf_sconcate_seg_wft.test.sh semi_crf_${args}.${pid}.params`

**Without Fine Tuning**

* `./run/ner_nsemicrf_sconcate_seg_woft.train.sh`
* `./run/ner_nsemicrf_sconcate_seg_woft.test.sh semi_crf_${args}.${pid}.params`

# Get help

Use `--help` option in the executable binaries to get more help.
Or write to Yijia Liu <oneplus.lau@gmail.com>.
