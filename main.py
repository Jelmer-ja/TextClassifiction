import sklearn as skl
import numpy as np
import scipy as sc
from classes import *
import nltk
from nltk.corpus import stopwords, brown
from nltk.corpus import wordnet as wn
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.probability import *
from scipy import stats
import pyphen
import re
import string
import copy
import numpy as np
import statsmodels.api as sm
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from collections import Counter
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing.dummy import Pool

"""
WORD LISTS/SETUP
"""
stop_words = set(stopwords.words('english'))
tags = ['CC','CD','DT','EX''FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT',
        'POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBP','VBZ','WDT','WP','WP$','WRB']
present_tenses_auxilliaries = ['am','is','are','has','have','do','does']
past_tenses_auxilliaries = ['was','were','been','had','did','done']
bees = ['am','is','are','were','be','being','been']
colors = ['Amber','Black','Blue','Brown','Burgundy','Chocolate','Coffee','Crimson','Emerald','Erin','Gold',
          'Gray','Green','Harlequin','Indigo','Ivory','Jade','Lavender','Olive','Orange','Raspberry','Red',
          'Rose','Ruby','Sapphire','Scarlet','Silver','Violet','Viridian','White','Yellow']
            #https://simple.wikipedia.org/wiki/List_of_colors which appeared in train set
gore = ['blood','flesh','bloody','bloodstained','mangled','liver','heart','brain','splatter','splattering','splattered',
        'carnage','slash','slashed','slashing','organ','slaughter','slaughtered','slaughtering'] #TODO good list
#cohesives = []
pronouns = ['I','my','me','mine','myself','you','your','yours','yourself','he','him','his','himself','she','her','hers','herself','it',
            'its','itself','they','them','theirs','yourselves','themselves']
causalverbs = ['make','made','cause','caused','allow','allowed','help','helped','have','had','enable','enabled','keep','kept',
               'hold','held','let','force','forced','require','required','making','causing','allowing','helping','having',
               'enabling','keeping','holding','letting','forcing','requiring']
causalparts = ['because','despite','resulting','thus','consequently','so','as','since']
hedgesndt = ['almost','maybe','somewhat','likely','barely','mildly','little','pretty','fairly']
amplifiers = ['completely','extremely','incredibly','quite','very','mostly','amazingly','really','definitely','exactly',
              'awfully']
negations = ['not','neither','nor','none','t','\'t','never','nobody','nowhere','no']
semper = ['seem','appear','seemed','appeared','seeming','appearing']
bow_list = [] #Total word list for bag of words features
dic = pyphen.Pyphen(lang='en')
print('Word lists created')
brownwords = FreqDist()
for sentence in brown.sents():
    for word in sentence:
        brownwords[word] += 1
print('Brown corpus loaded')

"""
FUNCTIONS
"""

def main():
    object = storydata(True)
    train_data = object.getTrain()
    test_data = object.getTest()
    print('Data imported to main class')

    # Extract the features
    train_X = [x[0] for x in train_data]
    train_y = [x[1] for x in train_data]
    test_X = [x[0] for x in test_data]
    test_y = [x[1] for x in test_data]

    #prepare_dictionary(train_X) #31984
    #prepare_dictionary(test_X)
    #dummy = wn.synsets('dummy')#quit()

    #pool = Pool(processes=8)
    features = []#map(extract_features, train_X)
    features = list(features)
    #pool.close()
    print('Features extracted\n')

    #Train and test the classifier
    #classificationAnalysis(train_X,train_y,test_X,test_y,features)

    #Apply regression
    regressionAnalysis(train_X,train_y,test_X,test_y,features)

def regressionAnalysis(train_X,train_y,test_X,test_y,features):
    #reg = LinearRegression()
    test_features = np.array(map(extract_features, test_X))
    print(str(len(test_features)) + ' ' + str(len(test_y)))
    # Train the model using the training sets
    #reg.fit(train_features, train_y)
    X = test_features
    y = test_y
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

def classificationAnalysis(train_X,train_y,test_X,test_y,features):
    # Classify and evaluate
    """
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split([str(i) for i in range(0, 620)], train_y)):
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


    #FINAL CLASSIFICATION RESULTS
    test_features = list(map(extract_features_bow, test_X))
    print('Test features extracted')
    y_pred = classify(features, train_y, test_features)
    recall2,precision2, f1 = evaluate(test_y, y_pred)
    print("Total test recall", recall2)
    print("Total test precision", precision2)
    print("Total test f-score", f1)

    """
    #Step two, view classification without certain features
    indices = [0,1,2,3,4,5,6,range(7,40),range(40,66),66,67,68,69,70,71,72,73,74,75,76,range(77,81),[81,82],[83,84]]
    test_features = list(map(extract_features, test_X))
    print('Test features extracted')
    for index in indices:
        train_features2 = copy.deepcopy(features)
        test_features2 = copy.deepcopy(test_features)
        if isinstance(index,list):
            index.reverse()
            for i in index:
                for j in range(0, 620):
                    train_features2[j].pop(i)
                    if (j < 206):
                        test_features2[j].pop(i)
        else:
            for j in range(0,620):
                train_features2[j].pop(index)
                if (j < 206):
                    test_features2[j].pop(index)

        y_pred = classify(train_features2, train_y, test_features2)
        recall2,precision2, f1 = evaluate(test_y, y_pred)
        print(index)
        print("Averaged total test recall", recall2)
        print("Averaged total test precision", precision2)
        print("Averaged total test f-score", f1)


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
    bag_of_words = [string.lower(x) for x in wordpunct_tokenize(text)]
    bag_of_sents = [string.lower(x) for x in sent_tokenize(text)]
    tokenizedsents = [wordpunct_tokenize(x) for x in bag_of_sents]
    pos_tags = nltk.pos_tag(bag_of_words)
    features = []
    """
    #LITERARY FEATURES
    """
    #ORWELL
    senlen = float(len(bag_of_words)) / len(bag_of_sents)
    features.append(senlen)  # Average sentence length, n = 1
    features.append(float(sum([len(x) for x in bag_of_words])) / len(bag_of_words)) #average word length, n = 2
    #TODO: Occurence of metaphors
    features.append(float(sum([1 for x in pos_tags if x[1][0] == 'V' and '-' not in dic.inserted(x[0])])) / len(bag_of_sents)) #Nr of monosyllabic verbs, nrmsl, n = 3

    #KING
    nr = 0
    for s in tokenizedsents: #Passive voice detection: Sentences are detected as passive if they contain a version #of the verb "to be" followed by any verb form but a gerund
        s_tags = nltk.pos_tag(s) #of the verb "to be" followed by any verb form but a gerund
        passive = False; be = False
        for t in s_tags:
            if(be and t[1][0] == 'V' and t[1] != 'VBG'):
                passive = True
            if (t[0] in bees):
                be = True
        if(passive):
            nr += 1
    features.append(float(nr) / len(bag_of_sents)) # Passive voice rate, nrmsl. Inspired by https://github.com/j-c-h-e-n-g/nltk-passive-voice/blob/gh-pages/passive.py, n = 4
    #features.append(float(sum([1 for x in bag_of_words if string.lower(x) in colors])) / len(bag_of_sents)) #Occurence of colors, nrmsl, n = 0
    #features.append(float(sum([1 for x in bag_of_words if string.lower(x) in gore])) / len(bag_of_sents)) #Occurence of gore, nrmsl, n = 0
    nr = 0
    for tokens in tokenizedsents:
        nr += len(tokens) - len(set(tokens))
    features.append(float(nr) / len(bag_of_sents)) #Word repetition within sentences, nrmsl, n = 5
    features.append(float(len([x for x in pos_tags if x[1] == 'RB' or x[1] == 'RBR' or x[1] == 'RBS'])) / len(bag_of_sents)) #Adverb usage
    # per sentence, nrmsl - Same as normal PoS-tagging?, n = 6
    features.append(float(len(bag_of_words)) / sum([1 for l in text if l == '\n'])) #Average paragraph length, n = 7

    """
    TEXT MINING FEATURES
    """
    for t in tags:
        features.append(float(len([x for x in pos_tags if x[1] == t])) / len(bag_of_words) * 100)  # PoS tag percentages, n = 33 + 7
    for t in string.ascii_lowercase:
        features.append(float(len([x for x in text if x == t])) / len(bag_of_words) * 100) #Letter frequency percentages 59 + 7
    features.append(float(sum([1 for x in bag_of_words if x in stop_words])) / len(bag_of_words) * 100)  # stop word percentage n = 60 + 7
    features.append(len(bag_of_words))  # nr of words = story length n = 61 + 7
    features.append(len(bag_of_sents))  # nr of sentences = story length in sentences, n = 62 + 7
    wordlengths = [len(x) for x in bag_of_words]
    counts = Counter(wordlengths)
    features.append(entropy(counts.values())) # entropy of word lengths, 63 + 7
    syllengths = [dic.inserted(x).count('-') for x in bag_of_words]
    awlength = float(sum(syllengths)) / len(bag_of_words)
    features.append(awlength) # avg word length by syllables, 64 + 7
    features.append(
        float(sum([1 for x in bag_of_words if (x in present_tenses_auxilliaries or x in past_tenses_auxilliaries)])) / len(
            bag_of_sents))  # Auxilliaries, nrmsl, n = 65 + 7
    """
    #ESSAY GRADING
    """
    features.append(float(sum([1 for x in bag_of_words if string.lower(x) in pronouns])) / len(bag_of_sents)) #Personal pronoun usage, nrmsl, 66 + 7
    features.append(float(sum([1 for x in bag_of_words if string.lower(x) in causalverbs]) / (sum([1 for x
        in bag_of_words if string.lower(x) in causalparts]) + 1))) #Causal verbs divided by causal particles 67 + 7
    features.append(float(len(set(bag_of_words))) / len(bag_of_words)) #Lexical diversity 68 + 7
    features.append(0.39 * senlen + 11.8 * awlength - 15.59) #Flesch-Kincaid level 69 + 7
    senses = [len(wn.synsets(x)) for x in bag_of_words]
    features.append(float(sum(senses)) / len(senses)) # Average nr of senses per word 70 + 7
    features.append(float(sum([1 for x in bag_of_words if string.lower(x) in hedgesndt])) / len(bag_of_sents)) #71 + 7
    features.append(float(sum([1 for x in bag_of_words if string.lower(x) in negations])) / len(bag_of_sents)) #72 + 7
    features.append(float(sum([1 for x in bag_of_words if string.lower(x) in amplifiers])) / len(bag_of_sents)) #73 + 7
    features.append(float(sum([1 for x in bag_of_words if string.lower(x) in semper])) / len(bag_of_sents)) #74 + 7
    fdist = FreqDist(bag_of_words)
    freqlist = [fdist.freq(x) for x in bag_of_words]
    features.append(entropy(freqlist)) #Entropy of word frequency within story 75 + 7
    features.append(float(sum(freqlist)) / len(bag_of_words)) #Average word frequency within story, 76 + 7
    brownlist = [brownwords[x] for x in bag_of_words]
    features.append(entropy(brownlist))  # Entropy of word frequency in brown corpus 77 + 7
    features.append(float(sum(brownlist)) / len(bag_of_words))  # Average word frequency in brown corpus, 78 + 7
    return features

def extract_features_bow(text):
    bag_of_words = [string.lower(x) for x in wordpunct_tokenize(text)]
    dict = {}
    for word in bow_list:
        dict[word] = 0
    for word in bag_of_words:
        dict[word] += 1
    return dict.values()

def extract_features_test(text):
    return [len(text)]

def prepare_dictionary(stories):
    #Create dictionary entries for every word in the data
    for s in stories:
        bag_of_words = [string.lower(x) for x in wordpunct_tokenize(s)]
        for w in bag_of_words:
            if w not in bow_list:
                bow_list.append(w)

# Classify using the features
def classify(train_features, train_labels, test_features):
    # TODO: (Optional) If you would like to test different how classifiers would perform different, you can alter
    # TODO: the classifier here.
    clf = DecisionTreeClassifier() #SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)

def printColors(stories):
    colorRates = [0 for i in range(0,len(colors))]
    for story in stories:
        for i in range(0,len(colors)):
            bag_of_words = [x for x in wordpunct_tokenize(story)]
            if colors[i] in bag_of_words:
                colorRates[i] += 1
    #Print which colors are used
    for i in range(0,len(colors)):
        print(colors[i] + ': ' + str(colorRates[i]))

if(__name__ == '__main__'):
    main()