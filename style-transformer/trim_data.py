# This model requires sentences to be a maximum of X = 80 tokens long
# Trim each line to be 80 tokens long

MAX_LEN = 80
DATASET = "novels"

sets = ["dev", "test", "train"]
classes = ["neg", "pos"]

def main():
    for x in sets:
        for cls in classes:
            in_filename = "data/{}/{}.{}".format(DATASET, x, cls)
            out_filename = "data/{}_short/{}.{}".format(DATASET, x, cls)

            print("Trimming %s to %s..." % (in_filename, out_filename))

            fin  = open(in_filename, "r", encoding="utf-8")
            fout = open(out_filename, "w", encoding="utf-8")

            for ln in fin:
                pcs = ln.split(" ")
                out_ln = pcs[:MAX_LEN].join(" ") + "\n"
                fout.write(out_ln)

            fin.close()
            fout.close()


if __name__ == '__main__':
    main()
