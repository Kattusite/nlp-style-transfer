import fasttext

def summarize(r):
    print(r[0], "training examples")
    print("%.2f %%" % (float(r[1]) * 100), "neg accuracy")
    print("%.2f %%" % (float(r[2]) * 100), "pos accuracy")

data = "../data/merged/{}_ft.txt"


# baseline: ~89-90% accs
# model = fasttext.train_supervised(data.format("train"))

# Current best:
# more epochs, more lr, trigrams: ~91.5% accs
model = fasttext.train_supervised(data.format("train"), lr=1.0, epoch=25, wordNgrams=3)

# more epochs, more lr, bigrams, hierarchichal softmax: ~ 90.7% accs (but faster)
# using 2grams acc ~90.5/90.6
# using 3grams acc ~90.5
# using 4grams acc drops to 89
# model = fasttext.train_supervised(data.format("train"), lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
model.save_model("evaluator/dickens_verne_fasttext.bin")

print("testing", data.format("dev"))
results = model.test(data.format("dev"))

print(results)

summarize(results)
