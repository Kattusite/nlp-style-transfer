import kenlm


model=kenlm.Model("dickens_verne_kenlm.bin")

tot  = 0
n = 0

with open("../data/novels_short/dev.combined") as f:
    for ln in f:
        tot += model.perplexity(ln)
        n += 1

ppl = tot / n

print(f"Tested KenLM LM against {n} sentences")
print(f"Resulting ppl: {ppl}")
