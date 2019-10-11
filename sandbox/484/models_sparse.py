from sentiment_data import SentimentExample
from collections import Counter
from typing import List
from utils import Indexer
from pprint import pprint
import random

import numpy as np

POSITIVE = 1
NEGATIVE = 0

PrintTop10 = True

PrintTraining = True
winsize = 100 # Size of sliding window over which to average losses
printEvery = 300

ShuffleSGD = True # Whether to randomize SGD iteration order

TrainingEpochs = 80 # 20   25
LearningRate = 5 # 0.1 (0.2 has ~76% dev) (0.5 too high)
L2Alpha = 0 # 0, 1e-6

# Feature extraction base type. Takes an example and returns an indexed list of features.
class FeatureExtractor(object):
    # Extract features. Includes a flag add_to_indexer to control whether the indexer should be expanded.
    # At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
    def extract_features(self, ex: SentimentExample, add_to_indexer: bool = False):
        raise Exception("Don't call me, call my subclasses")


# Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, ex, add_to_indexer=False):
        features = Counter()
        for w in ex.words:
            feat_idx = self.indexer.add_and_get_index(w) if add_to_indexer else self.indexer.index_of(w)
            if feat_idx != -1:
                features[feat_idx] += 1.0
        return features


# Bigram feature extractor analogous to the unigram one.
class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, ex, add_to_indexer=False):
        features = Counter()
        # Loop over words w1 from 0 to n-2 and words w2 from 1 to n-1
        n = len(ex.words) - 1
        for i in range(n):
            w1 = ex.words[i]
            w2 = ex.words[i+1]
            bg = (w1, w2)
            feat_idx = self.indexer.add_and_get_index(bg) if add_to_indexer else self.indexer.index_of(bg)
            if feat_idx != -1:
                features[feat_idx] += 1.0
        return features



# Use your creativity and propose a better feature extractor here!
class CustomizedFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")

    def extract_features(self, ex, add_to_indexer=False):
        raise Exception("Must be implemented")


# Sentiment classifier base type
class SentimentClassifier(object):
    # Makes a prediction for the given
    def predict(self, ex: SentimentExample):
        raise Exception("Don't call me, call my subclasses")


# Always predicts the positive class
class AlwaysPositiveClassifier(SentimentClassifier):
    def predict(self, ex: SentimentExample):
        return 1

###############################################################################
#                               NAIVE BAYES
#
###############################################################################

# Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
# superclass
class NaiveBayesClassifier(SentimentClassifier):
    def __init__(self, priors, probs, feat_extractor):

        self.priorProb = priors
        self.wcProb = probs
        self.feat_extractor = feat_extractor

    def predict(self, ex):

        features = self.feat_extractor.extract_features(ex)
        posProb = self.priorProb[POSITIVE]
        for w in features:
            posProb *= (self.wcProb[POSITIVE][w] ** features[w])

        negProb = self.priorProb[NEGATIVE]
        for w in features:
            negProb *= (self.wcProb[NEGATIVE][w] ** features[w])

        if posProb > negProb:
            return 1
        else:
            return 0



# Train a Naive Bayes model on the given training examples using the given FeatureExtractor
# TODO: Use log probabilities to prevent underflow
def train_nb(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> NaiveBayesClassifier:
    alpha = 1
    add_to_indexer = True

    # Step 1: Compute vocabulary V of all words in train_exs
    # We don't actually need this explicitly
    #V = list(set.union(*[set(ex.words) for ex in train_exs]))


    # Step 2: Compute count(positive) / # docs, count(negative) / # docs
    nDocs = len(train_exs)
    nClass = [
        0, # How many docs classed as negative
        0  # How many docs classed as positive
    ]
    for ex in train_exs:
        nClass[ex.label] += 1

    priorProbs = [
        nClass[NEGATIVE] / nDocs, # Prior prob. of seeing a negative example
        nClass[POSITIVE] / nDocs, # Prior prob. of seeing a pos. example.
    ]

    # Step 3a: Compute count(w,c) for all w, c
    # For each doc d:
    # NOTE: I say "word i" but really mean "feature i", as features may be bigrams, etc
    classCounts = [
        Counter(), # How many times has word i appeared in neg class?
        Counter(), # How many times has word i appeared in pos class?
    ]
    for ex in train_exs:
        # Count word occurrences in this example.
        features = feat_extractor.extract_features(ex, add_to_indexer)

        # Merge these word occurrences with the counter for the appropriate class
        classCounts[ex.label] += features

    V = len(feat_extractor.indexer)

    # Step 3b: Add alpha smoothing is carried out in step c
    # Step 3c: Calculate P(w | c)

    wcProbs = [
        [],
        []
    ]
    # Compute P(w | -) for each w
    # Compute P(w | +) for each w
    for cls in [NEGATIVE, POSITIVE]:
        for w in range(V):
            num = classCounts[cls][w] + alpha
            den = sum(classCounts[cls].values()) + (alpha * V)
            wcProbs[cls].append(num / den)

    # Print 10 words with largest P(w | +) / P(w | -) and vice versa
    if PrintTop10:
        score = lambda w : wcProbs[POSITIVE][w] / wcProbs[NEGATIVE][w]
        word = lambda w : feat_extractor.indexer.get_object(w)
        n = len(feat_extractor.indexer)
        scores = [(word(w), score(w)) for w in range(n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        print("\nMost positive words:")
        pprint(scores[:10])
        print("\nMost negative scores:")
        pprint(scores[-10:])


    return NaiveBayesClassifier(priorProbs, wcProbs, feat_extractor)


###############################################################################
#                            LOGISTIC REGRESSION
#
###############################################################################


# Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
# superclass
class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, w, b, feat_extractor):
        self.w = w
        self.b = b
        self.feat_extractor = feat_extractor

    def predict(self, ex):
        features = self.feat_extractor.extract_features(ex)
        x = npFromCounter(features, len(self.w))
        z = wxb(self.w, x, self.b)
        yh = sigmoid(z)
        return round(yh)

# Given a counter c, create an equivalent numpy vector of length n
def npFromCounter(c, n):
    x = np.zeros(n)
    for k in c:
        x[k] = c[k]
    return x

# Given a counter c create a numpy array of the indices represented in c
# i.e. those we'd need to update
def npIndFromCounter(c):
    x = np.fromiter(c.keys(), int)
    return x

# Given a counter c create a numpy array of the values represented in c
# i.e. a sparse value vector
def npValFromCounter(c):
    x = np.fromiter(c.values(), float)
    return x

def wxb(w, x, b):
    return np.asscalar(w.dot(x) + b)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Given a true class y and predicted class yh return the cross entropy loss
def celoss(y, yh):
    if y == 1:
        return -np.log(yh)
    else:
        return -np.log(1-yh)

# Train a Perceptron model on the given training examples using the given FeatureExtractor
def train_lr(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, lr=LearningRate, alpha=L2Alpha) -> LogisticRegressionClassifier:
    learning_rate = lr

    print("training logistic regression with LR", learning_rate, "alpha", alpha)

    # Figure out how many weights we need (# distinct features)
    features = Counter()
    for ex in train_exs:
        fs = feat_extractor.extract_features(ex, True)
        features += fs

    n = len(features)
    print(n, "dimensions")

    # Randomly initialize weights and biases
    w = np.random.random(n)
    b = np.random.random(1)

    # For each training tuple (xi, yi):
    i = 0 # Number of training examples seen (for printing loss)
    for epoch in range(TrainingEpochs):
        if ShuffleSGD:
            random.shuffle(train_exs)

        losses = []
        epochLosses = []
        for ex in train_exs:

            # Step 1a: Compute yh_i
            features = feat_extractor.extract_features(ex, False) # True or false?
            ii = npIndFromCounter(features) # the indexes that need updates
            #x = npFromCounter(features, n)
            z = wxb(w, x, b)
            yh = sigmoid(z) # in 0,1
            y = ex.label

            if PrintTraining:
                # Step 1b: Compute loss L(yh, y)
                loss = celoss(y, yh)
                epochLosses.append(loss)
                losses.append(loss)
                losses = losses[-winsize:] # keep sliding window of last winsize losses

                # if i % printEvery == 0:
                #     avgLoss = sum(losses) / winsize
                #     print("avg loss: ", avgLoss)

            # Step 2: Compute the gradient of Loss wrt parameters.
            dy = yh - y
            gradW = dy * x[ii]
            gradB = dy

            # Step 3: Gradient descent. Update parameters.
            w[ii] -= learning_rate * gradW
            b     -= learning_rate * gradB

            l2w = 2 * learning_rate * alpha * w[ii]
            l2b = 2 * learning_rate * alpha * b
            w[ii] -= l2w
            b     -= l2b

            i += 1

        # Report the average loss over the entire epoch
        if PrintTraining:
            epochLoss = sum(epochLosses) / len(epochLosses)
            print("epoch", epoch, "\tloss: ", epochLoss)

    return LogisticRegressionClassifier(w, b, feat_extractor)



# Main entry point for your modifications. Trains and returns one of several models depending on the args
# passed in from the main method.
def train_model(args, train_exs):
    # Initialize feature extractor
    if args.feature == "unigram":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feature == "bigram":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feature == "customized":
        # Add additional preprocessing code here
        feat_extractor = CustomizedFeatureExtractor(Indexer())
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Train the model
    if args.model == "AlwaysPositive":
        model = AlwaysPositiveClassifier()
    elif args.model == "NaiveBayes":
        model = train_nb(train_exs, feat_extractor)
    elif args.model == "LogisticRegression":
        model = train_lr(train_exs, feat_extractor)
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --feature")
    return model
