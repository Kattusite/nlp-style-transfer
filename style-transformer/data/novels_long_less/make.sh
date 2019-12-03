#!/usr/bin/env bash

# actually this script isn't used - use trim_data.py instead

for FILE in train.neg train.pos dev.neg dev.pos test.neg test.pos
do
  head -n 20000 ../novels/$FILE > $FILE
done
