import sys, random

# First argument should be the label to apply
# (Perhaps an integer 1 or 0), perhaps "dickens" or "wilde"
# Remaining args are filenames to check

# How much data to reserve for training, testing
TRAIN = 0.75
DEV = 0.15
TEST = 0.10

def main():

    sentences = []

    author = ""

    label = sys.argv[1]
    filenames = sys.argv[2:]
    for filename in filenames:

        pcs = filename.split("/")
        author = pcs[1]

        book = pcs[-1]
        print("\n\n\n================= %s ===============" % book)

        # Open the input file
        f = open(filename, "r", encoding="utf-8")

        lines = [ln for ln in f]
        sentences += lines
        print(len(lines), "lines")

        f.close()

    # "Randomly" shuffle the dataset.
    # Want my partitioning to be reproducible
    random.seed(42)
    random.shuffle(sentences)

    # Take the first portion for training, then for dev, then test
    N = len(sentences)
    trainN = int(TRAIN * N)
    devN = int((TRAIN + DEV) * N)

    datasets = {
        "train": sentences[:trainN],
        "dev": sentences[trainN:devN],
        "test": sentences[devN:]
    }

    filename = "labeled/{}/{}.txt"


    for name, data in datasets.items():
        outfile = open(filename.format(author, name), "w", encoding="utf=8")

        for sentence in data:
            outfile.write("{}\t{}".format(label, sentence))
        outfile.close()



if __name__ == '__main__':
    main()
