from matplotlib import pyplot as plt
import numpy as np
import re

iters = [i for i in range(25, 276, 25)]

trial = "Dec04022032"
filename = "save/{}/eval_log.txt".format(trial)

float_re = "{}:\s+([0-9]+\.[0-9]+)"

data_names = [
    "iter",
    "acc_pos",
    "acc_neg",
    "bleu_pos",
    "bleu_neg",
    "ppl_pos",
    "ppl_neg"
]

def pprint(ls):
    for x in ls:
        print(x)

# Build the re to capture each data point
data_res = ["iter\s+(\d+):"] + [float_re.format(nm) for nm in data_names[1:]]
data_re = "\s+".join(data_res)

print(data_re)

# initialize empty data arrays
data = {}
for nm in data_names:
    data[nm] = []

with open(filename, "r") as f:
    for ln in f:
        ln = ln.strip()
        if ln == "":
            continue

        match = re.search(data_re, ln)
        for i, nm in enumerate(data_names):
            data[nm].append(float(match.group(i+1)))

# pprint(data["iter"])
# pprint(data["bleu_pos"])

# pos = dickens (to verne)
# neg = verne (to dickens)


# acc, acc_ax = plt.subplots()
# acc_ax.set_title("Style Accuracy over Time")
# acc_ax.plot(data["iter"], data["acc_pos"], label="Dickens to Verne")
# acc_ax.plot(data["iter"], data["acc_neg"], label="Verne to Dickens")
# acc_ax.set_xlabel("iteration")
# acc_ax.set_ylabel("accuracy")


color1 = "#63d297"
# color1 = "#4ba173"
# color2 = "#ffd9b3"
color2 = "#CFEE9E"

plt.figure()
plt.title("Style Transfer Accuracy over Time")
plt.plot(data["iter"], data["acc_pos"], label="Dickens to Verne", color=color1)
plt.plot(data["iter"], data["acc_neg"], label="Verne to Dickens", color=color2)
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.legend()

plt.figure()
plt.title("Semantic Preservation over Time")
plt.plot(data["iter"], data["bleu_pos"], label="Dickens to Verne", color=color1)
plt.plot(data["iter"], data["bleu_neg"], label="Verne to Dickens", color=color2)
plt.xlabel("iteration")
plt.ylabel("BLEU")
plt.legend()

plt.figure()
plt.title("Perplexity over Time")
plt.plot(data["iter"], data["ppl_pos"], label="Dickens to Verne", color=color1)
plt.plot(data["iter"], data["ppl_neg"], label="Verne to Dickens", color=color2)
plt.xlabel("iteration")
plt.ylabel("perplexity")
plt.legend()


#
# bleu = plt.figure()
#
#
# ppl = plt.figure()
#
# plt.plot(iters, d_to_v, label="Dickens to Verne")
# plt.plot(iters, v_to_d, label="Verne to Dickens")
# plt.legend()
# plt.xlabel("training iteration")
# plt.ylabel("BLEU")
# plt.title("BLEU score over time")
#
plt.show()
