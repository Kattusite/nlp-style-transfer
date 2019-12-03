# Tokenize the input sentences to split out punctuation markings
# Also remove named entities

import sys, re
import nltk.data

REPLACE = True # Whether or not we should replace named entities with token or just delete

# https://stackoverflow.com/questions/43742956/fast-named-entity-removal-with-nltk
def extract_nonentities(tree):
    tokens = [leaf[0] for leaf in tree if type(leaf) != nltk.Tree]
    return " ".join(tokens) # TODO: Not necessarily rigorous

def replace_nonentities(tree):
    """Given a tree, replace all named entities with a placeholder token, like
    <NAMED_ENTITY>"""
    placeholder = "<NAME>" # "<NAMED_ENTITY>"

    tokens = []
    for leaf in tree:
        if type(leaf) != nltk.Tree:
            tokens.append(leaf[0])
        else:
            tokens.append(placeholder)

    return " ".join(tokens)

def main():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

    single_quote_re = "(``|“|”|'')"
    double_quote_re = "[‘’`]"

    filenames = sys.argv[1:]
    for filename in filenames:

        pcs = filename.split("/")
        pcs[0] = "tokenized"
        outFilename = "/".join(pcs)
        book = pcs[-1]
        print("\n================= %s ===============" % book)

        # Open the input and output files
        f = open(filename, "r", encoding="utf-8")
        out = open(outFilename, "w", encoding="utf-8")

        tokenized_lines = [nltk.word_tokenize(ln) for ln in f]

        tagged = nltk.pos_tag_sents(tokenized_lines)
        chunked = nltk.ne_chunk_sents(tagged)

        nonentities = []
        for tree in chunked:

            x = None
            if REPLACE:
                x = replace_nonentities(tree)
            else:
                x = extract_nonentities(tree)

            nonentities.append(x)

        # TODO: replace weird quotes with " or '

        # ne = ne.replace('``', '"')
        # ne = ne.replace('“', '"')
        # ne = ne.replace('”', '"')
        # ne = ne.replace("''", '"')
        # ne = ne.replace("’", "'")
        # ne = ne.replace("‘", "'")
        ne = re.sub(double_quote_re, '"', ne)
        ne = re.sub(single_quote_re, "'", ne)

        for ne in nonentities:
            out.write(ne + "\n")
            # out.write("\n-------------\n")


        # Close the input and output files
        f.close()
        out.close()


if __name__ == '__main__':
    main()
