# This model requires sentences to be a maximum of X = 80 tokens long
# Trim each line to be 80 tokens long
# Actually trim a little shorter for safety buffer

N_LONG = 20000 # number of examples for shorter dataset of larger examples
MAX_LEN_LONG = 60 # max for the shorter dataset of longer examples
MAX_LEN_LONG -= 1 # safety buffer

MAX_LEN = 16
MAX_LEN -= 1 # Safety buffer (e.g. for end token?)

DATASET = "novels"

sets = ["dev", "test", "train"]
classes = ["neg", "pos"]

def main():
    print("Trimming sentences longer than", MAX_LEN, "tokens")

    for x in sets:
        for cls in classes:

            lines_trimmed = 0

            in_filename = "data/{}/{}.{}".format(DATASET, x, cls)
            out_filename = "data/{}_short/{}.{}".format(DATASET, x, cls)
            out_ll_filename = "data/{}_long_less/{}.{}".format(DATASET, x, cls)

            print("Trimming %s to %s..." % (in_filename, out_filename))

            fin  = open(in_filename, "r", encoding="utf-8")
            fout = open(out_filename, "w", encoding="utf-8")
            fllout = open(out_ll_filename, "w", encoding="utf-8")

            for i, ln in enumerate(fin):
                pcs = ln.split()

                # Write out a smaller number of longer lines
                if i < N_LONG:
                    out_ln = " ".join(pcs[:MAX_LEN_LONG]) + "\n"
                    fllout.write(out_ln)

                # drop long examples entirely...
                if len(pcs) > MAX_LEN:
                    lines_trimmed += 1
                    continue

                out_ln = " ".join(pcs[:MAX_LEN]) + "\n"
                fout.write(out_ln)



            fin.close()
            fout.close()

            print(lines_trimmed, "lines trimmed.")


if __name__ == '__main__':
    main()
