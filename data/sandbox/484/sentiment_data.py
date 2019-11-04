class SentimentExample:
    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()


# Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences.
def read_sentiment_examples(infile):
    f = open(infile, encoding='iso8859')
    exs = []
    for line in f:
            fields = line.strip().split(" ")
            label = 0 if "0" in fields[0] else 1
            exs.append(SentimentExample(fields[1:], label))
    f.close()
    return exs
