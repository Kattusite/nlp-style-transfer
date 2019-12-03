from matplotlib import pyplot as plt

iters = [i for i in range(25, 276, 25)]


"""
iter   25:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter   50:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter   75:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  100:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  125:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  150:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  175:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  200:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  225:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  250:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
iter  275:  acc_pos: 1.0000 acc_neg: 0.0000 bleu_pos:  bleu_neg:  ppl_pos: -1.0000 ppl_neg: -1.0000
"""

# Bleu scores

# "pos" examples are dickens
d_to_v = [
    92.6741,
    94.1942,
    94.7160,
    95.1086,
    95.3472,
    95.3892,
    95.1138,
    95.5022,
    90.2218,
    95.8318,
    95.7779,
]

v_to_d = [
    90.9685,
    93.1366,
    94.0529,
    94.7508,
    94.6202,
    94.3635,
    92.6193,
    93.0469,
    92.4359,
    95.0889,
    96.4406,
]

plt.plot(iters, d_to_v, label="Dickens to Verne")
plt.plot(iters, v_to_d, label="Verne to Dickens")
plt.legend()
plt.xlabel("training iteration")
plt.ylabel("BLEU")
plt.title("BLEU score over time")

plt.show()
