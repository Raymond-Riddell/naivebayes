# naivebayes.py
"""Perform document classification using a Naive Bayes model."""

import argparse
import os
import pdb
import random

import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description="Use a Naive Bayes model to classify text documents.")
parser.add_argument('-x', '--training_data',
                    help='path to training data file, defaults to ROOT/trainingdata.txt',
                    default=os.path.join(ROOT, 'trainingdata.txt'))
parser.add_argument('-y', '--training_labels',
                    help='path to training labels file, defaults to ROOT/traininglabels.txt',
                    default=os.path.join(ROOT, 'traininglabels.txt'))
parser.add_argument('-xt', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testingdata.txt',
                    default=os.path.join(ROOT, 'testingdata.txt'))
parser.add_argument('-yt', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testinglabels.txt',
                    default=os.path.join(ROOT, 'testinglabels.txt'))
parser.add_argument('-n', '--newsgroups',
                    help='path to newsgroups file, defaults to ROOT/newsgroups.txt',
                    default=os.path.join(ROOT, 'newsgroups.txt'))
parser.add_argument('-v', '--vocabulary',
                    help='path to vocabulary file, defaults to ROOT/vocabulary.txt',
                    default=os.path.join(ROOT, 'vocabulary.txt'))


def main(args):
    accuracies = []
    alphas = [.00001, .04218, .22193, .01732, .00281, .01231, .00153, .03714, .95710, .04619, .54917, .00174, .27109,
              .00693, .00420, .92013, .02639, .00232, .00031, .00009]

    alphas.sort()

    print("Document Classification using Naïve Bayes Classifiers")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    newsgroups_path = os.path.expanduser(args.newsgroups)
    vocabulary_path = os.path.expanduser(args.vocabulary)

    # Load data from relevant files
    # ***MODIFY CODE HERE***
    print("Loading training data...")
    xtrain = np.loadtxt((os.path.join(ROOT, "trainingdata.txt")), dtype=int, delimiter=' ')

    print("Loading training labels...")
    ytrain = np.loadtxt(os.path.join(ROOT, "traininglabels.txt"), dtype=int)

    print("Loading testing data...")
    xtest = np.loadtxt(os.path.join(ROOT, "testingdata.txt"), dtype=int, delimiter=' ')

    print("Loading testing labels...")
    ytest = np.loadtxt(os.path.join(ROOT, "testinglabels.txt"), dtype=int)

    print("Loading newsgroups...")
    newsgroups = np.loadtxt(os.path.join(ROOT, "newsgroups.txt"), dtype=str)

    print("Loading vocabulary...")
    vocabulary = np.loadtxt(os.path.join(ROOT, "vocabulary.txt"), dtype=str)

    # Change 1-indexing to 0-indexing for labels, docID, wordID

    # Extract useful parameters
    num_training_documents = len(ytrain)
    num_testing_documents = len(ytest)
    num_words = len(vocabulary)
    num_newsgroups = len(newsgroups)

    for x in range(num_training_documents):
        ytrain[x] = ytrain[x] - 1

    for x in range(num_testing_documents):
        ytest[x] = ytest[x] - 1

    for x in range(len(xtrain)):
        xtrain[x][0] = xtrain[x][0] - 1
        xtrain[x][1] = xtrain[x][1] - 1

    for x in range(len(xtest)):
        xtest[x][0] = xtest[x][0] - 1
        xtest[x][1] = xtest[x][1] - 1

    for y in range(1):
        print("\n=======================")
        print("TRAINING")
        print("=======================")

        # Estimate the prior probabilities
        # GIVEN this dataset, what are the odds of any given document being any given class?
        print("Estimating prior probabilities via MLE...")
        from collections import Counter

        # Scan through the labels, getting the total number of times each label appears
        num_of_each = Counter(ytrain)
        prob_of_each = [None] * num_newsgroups

        for x in range(0, num_newsgroups):
            prob_of_each[x] = num_of_each[x] / num_training_documents

        priors = prob_of_each

        # Estimate the class conditional probabilities
        # Given one of these class types, prob of seeing data we have?
        print("Estimating class conditional probabilities via MAP...")

        total = 0
        indices = [0]

        for x in range(1, num_newsgroups):
            indices.append((total + num_of_each[x - 1]) - 1)
            total += num_of_each[x - 1]

        cc = numpy.zeros((num_words, num_newsgroups))

        for x in range(len(xtrain)):
            cc[xtrain[x, 1], ytrain[xtrain[x, 0]]] += xtrain[x, 2]

        sum_of_columns = cc.sum(axis=0)
        cc += alphas[y]

        class_conditionals = numpy.divide(cc, sum_of_columns)

        print("\n=======================")
        print("TESTING")
        print("=======================")

        # Test the Naive Bayes classifier
        print("Applying natural log to prevent underflow...")
        log_priors = np.log(priors)

        log_class_conditionals = np.log(class_conditionals)

        print("Counting words in each document...")
        # ***MODIFY CODE HERE***
        counts = np.zeros((num_testing_documents, num_words))

        for x in range(len(xtest)):
            counts[xtest[x, 0], xtest[x, 1]] += xtest[x, 2]

        print("Computing posterior probabilities...")

        log_posterior = np.matmul(counts, log_class_conditionals) + log_priors

        print("Assigning predictions via argmax...")
        pred = []
        for x in range(num_testing_documents):
            pred.append(np.argmax(log_posterior[x]))

        print("\n=======================")
        print("PERFORMANCE METRICS")
        print("=======================")

        # Compute performance metrics
        accuracy = 0
        for x in range(num_testing_documents):
            if pred[x] == ytest[x]:
                accuracy += 1

        accuracies.append(accuracy)

    # plt.xlabel("Alphas")
    # plt.ylabel("Accuracies")
    # plt.semilogx(alphas, accuracies)
    # plt.show()

    print("Accuracy: " + str(accuracy / num_testing_documents))
    cm = confusion_matrix(ytest, pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=newsgroups)
    disp.plot()
    plt.show()

    # pdb.set_trace()  # uncomment for debugging, if needed


if __name__ == '__main__':
    main(parser.parse_args())
