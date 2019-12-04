#!/usr/bin/env bash


# not runnable as-is, need to update the paths to lmplz and
# text files, just an indicator of how to make.

bin/lmplz -o 5 <text >text.arpa

export DATASETS=~/nlp-style-transfer/data/tokenized
cat $DATASETS/dickens/*.txt $DATASETS/verne/*.txt | ~/kenlm/build/bin/lmplz -o 5 >dickens_verne_kenlm.arpa
