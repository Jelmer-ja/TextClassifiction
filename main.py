import sklearn as skl
import numpy as np
import scipy as sc
from classes import *
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import re
import string
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn.svm import SVC

stop_words = set(stopwords.words('english'))
tags = ['CC','CD','DT','EX''FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT',
        'POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBP','VBZ','WDT','WP','WP$','WRB']

def main():
    object = storydata()
    train_data = object.getTrain()
    print('Data imported to main class')
    #test_data = object.getTest()

    # Extract the features
    train_X = [x[0] for x in train_data]
    train_y = [x[1] for x in train_data]

    features = list(map(extract_features, train_X))
    print('Features extracted\n')

    # Classify and evaluate
    skf = sklearn.model_selection.KFold(n_splits=10)
    scores = []
    for fold_id, (train_indexes, validation_indexes) in enumerate(
            skf.split(train_y, train_X)):
        # Print the fold number
        print("Fold %d" % (fold_id + 1))

        # Collect the data for this train/validation split
        train_features = [features[x] for x in train_indexes]
        train_labels = [train_y[x] for x in train_indexes]
        validation_features = [features[x] for x in validation_indexes]
        validation_labels = [train_y[x] for x in validation_indexes]

        # Classify and add the scores to be able to average later
        y_pred = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))

        # Print a newline
        print("")

    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")

# Evaluate predictions (y_pred) given the ground truth (y_true)
def evaluate(y_true, y_pred):
    # TODO: What is being evaluated here and what does it say about the performance? Include or change the evaluation
    # TODO: if necessary.
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score

#Use these functions to extract features out of the pastas
def extract_features(text):
    bag_of_words = [x for x in wordpunct_tokenize(text)]
    bag_of_sents = [x for x in sent_tokenize(text)]
    pos_tags = nltk.pos_tag(bag_of_words)

    features = []
    """
    TEXT MINING FEATURES
    """
    features.append(len(bag_of_words)) #nr of words
    features.append(len(bag_of_words) / len(bag_of_sents)) #avg nr of words per sentence
    features.append(len(bag_of_sents)) #nr of sentences
    features.append(len([x for x in bag_of_sents if x[-1] == '?'])) #nr of questions
    features.append(sum([len(x) for x in bag_of_words]) / len(bag_of_words)) #average word length
    for char in string.ascii_lowercase:
        features.append(len([x for x in text if x.lower() == char])) #character frequencies
    for t in tags:
        features.append(len([x for x in pos_tags if x[1] == t])) #PoS tags

    """
    LITERARY FEATURES
    """

    # TODO: Follow the instructions in the assignment and add your own features.
    return features

# Classify using the features
def classify(train_features, train_labels, test_features):
    # TODO: (Optional) If you would like to test different how classifiers would perform different, you can alter
    # TODO: the classifier here.
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)


"""
OLD CODE:

def sentimentTrain():
    return [('I love this sandwich.', 'pos'),
    ('this is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('he is my sworn enemy!', 'neg'),
    ('my boss is horrible.', 'neg')]

def sentimentTest():
    return [
    ('the beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')]

def classifierTest():
    train = sentimentTrain()
    test = sentimentTest()
    nbayes = cl.NaiveBayesClassifier(train)
    dtree = cl.DecisionTreeClassifier(train)
    maxent = cl.MaxEntClassifier(train)
    for i in range(0,6):
        print(str(nbayes.classify(test[i][0])))
    print(nbayes.accuracy(test))
    print(dtree.accuracy(test))
    print(maxent.accuracy(test))
"""
if(__name__ == '__main__'):
    main()