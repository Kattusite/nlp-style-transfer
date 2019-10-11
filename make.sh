#!/usr/bin/bash

# This file runs all of the data processing steps,
# from the raw data to the labeled datasets

# Remove formatting and metadata from ebooks
python cleaner.py data/dickens/*.txt
python cleaner.py data/wilde/*.txt

# Split the ebooks one sentence per line
python sentencer.py clean/dickens/*.txt
python sentencer.py clean/wilde/*.txt

# Aggregate data by author and assign it labels
python labeler.py 1 sentences/dickens/*.txt
python labeler.py 0 sentences/wilde/*.txt

# Aggregate data across authors, creating train/dev/test sets
python merger.py labeled/dickens/*.txt labeled/wilde/*.txt
