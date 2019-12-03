#!/usr/bin/env bash

# This file runs all of the data processing steps,
# from the raw data to the labeled datasets
# Usage:
# ./make.sh dickens wilde
# ./make.sh dickens verne

echo "Author 1: $1 \t Author 0: $2"

if [[ -z $1 ]] || [[ -z $2 ]]; then
  echo "Usage: ./make.sh <pos author(dickens)> <neg author(verne)>"
  exit
fi

# Remove formatting and metadata from ebooks
python cleaner.py raw/$1/*.txt
python cleaner.py raw/$2/*.txt

# Split the ebooks one sentence per line
python sentencer.py clean/$1/*.txt
python sentencer.py clean/$2/*.txt

# Tokenize and remove named entities
python tokenizer.py sentences/$1/*.txt
python tokenizer.py sentences/$2/*.txt

# Aggregate data by author and assign it labels
python labeler.py 1 tokenized/$1/*.txt
python labeler.py 0 tokenized/$2/*.txt

# Aggregate data across authors, creating train/dev/test sets
python merger.py labeled/$1/*.txt labeled/$2/*.txt

# Create output data for the style-transformer
for dataset in "train" "test" "dev"
do
  cut --complement -c 1-2 labeled/$1/$dataset.txt > ../style-transformer/data/novels/$dataset.pos
  cut --complement -c 1-2 labeled/$2/$dataset.txt > ../style-transformer/data/novels/$dataset.neg
done
