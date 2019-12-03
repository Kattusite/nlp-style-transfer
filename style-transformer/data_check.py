from torchtext import data
import csv

fs = [
    "train.pos",
    "train.neg",
    "test.pos",
    "test.neg",
    "dev.neg",
    "dev.pos"
]

for fn in fs:
    nm = "data/novels_short/" + fn

    ds = data.TabularDataset(
        path="data/novels_short/train.pos",
        format='tsv',
        fields=[("text", data.Field(batch_first=True, eos_token='<eos>'))],
        csv_reader_params={
            "quoting": csv.QUOTE_NONE,
        }
    )

    for d in ds:
        if len(d.text) > 35:
            print(d.text)
            break
