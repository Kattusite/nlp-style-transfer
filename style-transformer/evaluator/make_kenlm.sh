#!/usr/bin/env bash


# not runnable as-is, need to update the paths to lmplz and
# text files, just an indicator of how to make.

#bin/lmplz -o 5 <text >text.arpa

#export DATASETS=~/nlp-style-transfer/data/tokenized
#cat $DATASETS/dickens/*.txt $DATASETS/verne/*.txt | ~/kenlm/build/bin/lmplz -o 5 >dickens_verne_kenlm.arpa

cd ~/kenlm/build/
export kenlm="$PWD"

cd ~/nlp-style-transfer/style-transformer/data/novels_short
export data="$PWD"

cd ~/nlp-style-transfer/style-transformer/evaluator

echo "Building KenLM LM using order 5 and training data..."
cat $data/train.* | $kenlm/bin/lmplz -o 5 > dickens_verne_kenlm.arpa

echo "Converting to binary model..."
$kenlm/bin/build_binary dickens_verne_kenlm.arpa dickens_verne_kenlm.bin


echo "Evaluating on dev set..."
python ppl.py

