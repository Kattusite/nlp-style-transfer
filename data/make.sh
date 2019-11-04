#!/usr/bin/bash

# This file runs all of the data processing steps,
# from the raw data to the labeled datasets
# Usage:
# ./make.sh dickens wilde
# ./make.sh dickens verne

echo "Author 1: $1 \t Author 0: $2"

# Remove formatting and metadata from ebooks
python cleaner.py data/$1/*.txt
python cleaner.py data/$2/*.txt

# Split the ebooks one sentence per line
python sentencer.py clean/$1/*.txt
python sentencer.py clean/$2/*.txt

# Tokenize and remove named entities
# python tokenizer.py sentences/$1/*.txt
# python tokenizer.py sentences/$2/*.txt

# Aggregate data by author and assign it labels
python labeler.py 1 sentences/$1/*.txt
python labeler.py 0 sentences/$2/*.txt

# Aggregate data across authors, creating train/dev/test sets
python merger.py labeled/$1/*.txt labeled/$2/*.txt
