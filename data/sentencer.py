import sys
import nltk.data


def main():
    sent_detector = nltk.load('tokenizers/punkt/english.pickle')

    filenames = sys.argv[1:]
    for filename in filenames:

        pcs = filename.split("/")
        pcs[0] = "sentences"
        outFilename = "/".join(pcs)
        book = pcs[-1]
        print("\n\n\n================= %s ===============" % book)

        # Open the input and output files
        f = open(filename, "r", encoding="utf-8")
        out = open(outFilename, "w", encoding="utf-8")


        text = f.read()
        sentences = sent_detector.tokenize(text.strip())

        for s in sentences:
            s = s.replace("\n", " ")
            s += "\n"

            ws = s.split()

            # Short sentences are too hard to classify and
            # don't have much interesting stylistic content
            if len(ws) < 5:
                continue

            # Remove sentences enclosed in [  ] - usually captions or other metainfo
            out.write(s)
            # out.write("\n-------------\n")


        # Close the input and output files
        f.close()
        out.close()


if __name__ == '__main__':
    main()
