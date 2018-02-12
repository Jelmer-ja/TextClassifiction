import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import textblob.classifiers as cl
import urllib2
import math
from scipy import stats
import numpy as np
import random
import string
from bs4 import BeautifulSoup
from bs4.element import Comment
import cfscrape as cfs

class storydata:
    def __init__(self,regression):
        self.reg = regression
        self.stories = []
        self.ratings = []
        self.urls = []
        self.limits = [7.2,8.3]
        self.goodstories = []
        self.avgstories = []
        self.badstories = []
        self.collectStories()
        self.sortStories()
        self.shuffle()

        #print(len(self.goodstories)) #277
        #print(len(self.avgstories))  #265
        #print(len(self.badstories))  #284
        print('Data loaded')



    #fill the list of urls with the creepypasta story lins from the text file
    def getUrls(self):
        redflags = ['pastas-indexed-category','discussion-post','wp-content','category-','category/','update',
                    'page/','page-','tag/','author/','creepypasta-','-creepypasta']

        #Open the file of website urls, filter out the ones that are not stories themselves
        #Add the rest to the list of urls
        urlfile = open('urls.txt','r')
        for url in urlfile.readlines():
            clean = True
            for flag in redflags:
                if(flag in url):
                    clean = False
            if(clean):
                self.urls.append(url[:-2])
        print(len(self.urls))

    def saveStories(self):
        for i in range(0,len(self.urls)):
            url = self.urls[i]
            print('Saving story ' + str(i) + ': ' + url)
            text,rating = self.getPage(url)
            filename = '/home/jelmer/Documents/pastadata/' + str(i) + '.txt'
            f = open(filename,'w')
            f.write(str(rating))
            f.write('\n')
            f.write(text)
            f.close()

    def toAscii(self):
        for i in range(0,len(self.ratings)):
            print('Saving story ' + str(i))
            text = self.stories[i].decode('utf-8')
            text2 = text.encode('ascii','ignore')
            rating = self.ratings[i]
            filename = '/home/jelmer/Documents/pastadata2/' + str(i) + '.txt'
            f = open(filename,'w')
            f.write(str(rating))
            f.write('\n')
            f.write(text2)
            f.close()

    # Returns the story from the url as plaintext and the rating as a float
    def getPage(self, url):
        # Retrieve the url and create a parser
        scraper = cfs.create_scraper()
        page = scraper.get(url).content
        soup = BeautifulSoup(page, 'html.parser')
        [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
        readable_text = soup.getText().encode('utf-8','ignore')  # Extract the text from the html and convert it to ASCII


        # Find and extract the rating
        rb = soup.find('strong')
        rating = float(rb.text.strip())  # strip() is used to remove starting and trailing

        # Find and extract the story's text
        count = False
        text = ''
        for s in readable_text.splitlines():
            # print(s)
            if not count:
                if ('Add this post to your list of favorites' in s):
                    count = True
            else:
                if ('Rate this item:' in s):
                    break
                else:
                    text += s + '\n'

        # Delete newlines at the end of the text so that they don't mess with the data later
        for i in range(1, 20):
            if text[-1] == '\n':
                text = text[:-1]
            else:
                break
        return text, rating

    #Retrieve every story and score for every url in the list and add them to the 'stories' and 'ratings' lists
    def collectStories(self):
        for i in range(0,826):
            filename = '/home/jelmer/Documents/pastadata2/' + str(i) + '.txt'
            f = open(filename, 'r')
            lines = f.readlines()
            self.ratings.append(float(lines[0]))
            self.stories.append(string.join(lines[1:],''))
            f.close()

    #Turn the ratings from a double to a class label
    def sortStories(self):
        if(self.reg):
            for i in range(0,826):
                if (self.ratings[i] >= self.limits[1]):
                    self.goodstories.append((self.stories[i], self.ratings[i]))
                elif (self.ratings[i] > self.limits[0]):
                    self.avgstories.append((self.stories[i], self.ratings[i]))
                else:
                    self.badstories.append((self.stories[i], self.ratings[i]))
        else:
            for i in range(0,826):
                if (self.ratings[i] >= self.limits[1]):
                    self.goodstories.append((self.stories[i], 2))
                elif (self.ratings[i] > self.limits[0]):
                    self.avgstories.append((self.stories[i], 1))
                else:
                    self.badstories.append((self.stories[i], 0))

    #Shuffle the stories so that they are distributed over the sets independent of time
    def shuffle(self):
        random.seed(111)
        random.shuffle(self.goodstories)
        random.seed(222)
        random.shuffle(self.avgstories)
        random.seed(333)
        random.shuffle(self.badstories)

    #Calculate the class limits
    def getLimits(self):
        sort = sorted(self.ratings)
        print(sort[274])
        print(sort[550])

    #Plot the distribution of ratings among stories
    def plotRatings(self):
        ys = [0 for i in range(0,11)]
        for r in self.ratings:
            ys[int(round(r))] += 1
        labels = ['0','1','2','3','4','5','6','7','8','9','10']
        xs = np.arange(len(labels))
        width = 1
        plt.bar(xs, ys, width, align='center')
        plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
        plt.yticks(ys)
        plt.savefig('netscore.png')

    def getTrain(self):
        train = self.goodstories[0:208] + self.avgstories[0:199] + self.badstories[0:213]
        random.seed(444)
        random.shuffle(train)
        return train

    def getTest(self):
        test = self.goodstories[208:277] + self.avgstories[199:265] + self.badstories[213:284]
        random.seed(555)
        random.shuffle(test)
        return test