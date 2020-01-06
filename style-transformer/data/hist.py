"""Usage:
$  cd data/novels
$  python ../hist.py
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import re, sys
from scipy import stats

n = 0
avg_len = 0
lens = []
for x in ["dev", "test", "train"]:
    for y in ["neg", "pos"]:
        with open(f"{x}.{y}", "r", encoding="utf-8") as f:
            for ln in f:
                n += 1
                sen_len = len(ln.split())
                avg_len += sen_len
                lens.append(sen_len)

avg_len /= n
median = np.median(lens)
min = np.min(lens)
max = np.max(lens)
p99 = np.percentile(lens, 99)

# default plotting parameters for use with novels dataset
dataset = "Novels"
nbins = 25
rng = (0,200)

# font = {#'family' : 'normal',
#         # 'weight' : 'bold',
#         'size'   : 14}
#
# matplotlib.rc('font', **font)
# plt.rcParams.update({'font.size': 12})

# custom parameters specified by cmdline arg
if len(sys.argv) > 1:
    if sys.argv[1] == "yelp":
        dataset = "Yelp"
        nbins = None
        rng = None

print(f"Read {n} sentences with avglen of {avg_len}, median {median}")
print(f"min: {min} max: {max}")
print(f"99th %: {p99}")

if p99 > 31:
    print("31 is the xth percentile:", stats.percentileofscore(lens, 31))


color1 = "#63d297"
# color1 = "#4ba173"
# color2 = "#ffd9b3"
color2 = "#CFEE9E"

plt.figure()
plt.title(f"Sentence Length Distribution in {dataset} dataset")
plt.hist(lens, bins=nbins, range=rng, color=color1)
plt.xlabel("sentence length")
plt.ylabel("num. occurrences")
# plt.legend()

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
