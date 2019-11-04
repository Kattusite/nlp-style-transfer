# Load in data in the format it is expected by vectorize_data, from a list of
# text files as provided in data/merged/*.txt

import numpy as np

# Given a filename, read that file into a list of labels and a parallel list of
# sentences, as english strings
def load_data(filename):

    labels = []
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        for ln in f:
            label = int(ln[:2])
            sentence = ln[2:]

            labels.append(label)
            sentences.append(sentence)

    return sentences, np.array(labels)

TRAIN_DEFAULT = "data/merged/train.txt"
DEV_DEFAULT = "data/merged/dev.txt"

def load_all_data(train_file=TRAIN_DEFAULT, dev_file=DEV_DEFAULT):
    train_data = load_data(train_file)
    dev_data = load_data(dev_file)

    return train_data, dev_data
