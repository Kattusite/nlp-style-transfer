# sentiment_classifier.py

import argparse
import time
from models import train_model
from sentiment_data import read_sentiment_examples

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


# Evaluates a given classifier on the given examples
def evaluate(classifier, exs):
    print_evaluation([ex.label for ex in exs], [classifier.predict(ex) for ex in exs])


# Prints accuracy comparing golds and predictions, each of which is a sequence of 0/1 labels.
def print_evaluation(golds, predictions):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" %
                        (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1

    print("Accuracy: %i / %i = %.2f %%" %
          (num_correct, num_total,
           num_correct * 100.0 / num_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPositive',
                        choices=['AlwaysPositive', 'NaiveBayes', 'LogisticRegression'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--train_file', type=str, default='data/train.txt',
                        help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_file', type=str, default='data/dev.txt',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_file', type=str, default='data/test.txt',
                        help='path to test set (you should not call this yourself)')
    args = parser.parse_args()
    print(args)

    train_exs = read_sentiment_examples(args.train_file)
    dev_exs = read_sentiment_examples(args.dev_file)
    n_pos = 0
    n_neg = 0
    for ex in train_exs:
        if ex.label == 1:
            n_pos += 1
        else:
            n_neg += 1
    print("%d train examples: %d positive, %d negative" % (len(train_exs), n_pos, n_neg))
    print("%d dev examples" % len(dev_exs))

    # Train and evaluate
    start_time = time.time()
    model = train_model(args, train_exs)
    print("===== Train Accuracy =====")
    evaluate(model, train_exs)
    print("===== Dev Accuracy =====")
    evaluate(model, dev_exs)
    if args.test_file is not None:
        test_exs = read_sentiment_examples(args.test_file)
        print("===== Test Accuracy =====")
        print("%d test examples" % len(test_exs))
        evaluate(model, test_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))
