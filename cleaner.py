import sys

def main():
    filenames = sys.argv[1:]

    for filename in filenames:

        pcs = filename.split("/")
        pcs[0] = "clean"
        outFilename = "/".join(pcs)
        book = pcs[-1]
        print("================= %s ===============" % book)

        # Open the input and output files
        f = open(filename, "r", encoding="utf-8")
        out = open(outFilename, "w", encoding="utf-8")

        #
        seenStart = False
        for i, line in enumerate(f):


            out.write(line)

        # Close the input and output files
        f.close()
        out.close()


if __name__ == '__main__':
    main()
