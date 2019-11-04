import sys


def main():

    filenames = sys.argv[1:]

    print("running stats")

    for fn in filenames:

        print("=========== %s ==========" % fn)

        f = open(fn, "r", encoding="utf-8")

        max = 0
        for ln in f:
            words = ln.split(" ")
            if len(words) > max:
                max = len(words)

            if len(words) > 270:
                print(len(words), ln)

        print("longest sentence: ", max, "words")


if __name__ == '__main__':
    main()
