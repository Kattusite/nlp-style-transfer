import sys, random

# Usage:
# python merger.py labeled/dickens/*.txt labeled/wilde/*.txt



def main():

    sentences = {
        "train": [],
        "dev": [],
        "test": [],
    }

    filenames = sys.argv[1:]
    for filename in filenames:

        pcs = filename.split("/")
        author = pcs[1]
        dataset = pcs[-1]
        print("\n\n\n================= %s %s ===============" % (author, dataset))

        # Open the input file
        f = open(filename, "r", encoding="utf-8")

        lines = [ln for ln in f]
        print(len(lines))

        # Strip off extension to get "train", "dev", or "test"
        dataset = dataset.replace(".txt", "")
        sentences[dataset] += lines

        f.close()

    # "Randomly" shuffle the dataset.
    # Want my partitioning to be reproducible
    random.seed(42)

    for name, data in sentences.items():
        random.shuffle(data)

        # Open the output file and write out the sentences
        filename = "merged/{}.txt".format(name)
        outfile = open(filename, "w", encoding="utf=8")
        for sentence in data:
            outfile.write(sentence)
        outfile.close()



if __name__ == '__main__':
    main()
