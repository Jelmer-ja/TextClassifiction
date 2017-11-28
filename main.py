import sklearn as skl
import numpy as np
import scipy as sc
import textblob.classifiers as cl
from classes import *
import nltk

def main():
    #Create data
    train = sentimentTrain()
    test = sentimentTest()
    nbayes = cl.NaiveBayesClassifier(train)
    dtree = cl.DecisionTreeClassifier(train)
    maxent = cl.MaxEntClassifier(train)
    data = storydata()

    #for i in range(0,6):
        #print(str(nbayes.classify(test[i][0])))
    #print(nbayes.accuracy(test))
    #print(dtree.accuracy(test))
    #print(maxent.accuracy(test))

#Use these functions to extract features out of the pastas
def pos_features(word):
    features = {}
    #for suffix in common_suffixes:
    #    features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features

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

if(__name__ == '__main__'):
    main()