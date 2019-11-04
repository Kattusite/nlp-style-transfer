# From: https://developers.google.com/machine-learning/guides/text-classification/step-3
# And step 4
import sys
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 270 #default 500 but thats overkill for our purposes
# 270 seemed good by checking manually with stats.py

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

# Given a filename, read that file into a list of labels and a parallel list of
# sentences, as english strings
def read_file(filename):

    labels = []
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        for ln in f:
            label = int(ln[:2])
            sentence = ln[2:]

            labels.append(label)
            sentences.append(sentence)

    return labels, sentences


#################################################################################


# My function. Read
def main():

    train_file = "data/merged/train.txt"
    dev_file = "data/merged/dev.txt"

    if len(sys.argv) >= 2:
        train_file = sys.argv[1]
    if len(sys.argv) >= 3:
        dev_file = sys.argv[2]

    train_labels, train_data = read_file(train_file)
    dev_labels, dev_data = read_file(dev_file)

    train_seq, dev_seq, wids = sequence_vectorize(train_data, dev_data)

    print(len(train_seq), len(dev_seq), len(wids))
    # print(train_seq, dev_seq, wids)

if __name__ == '__main__':
    main()
