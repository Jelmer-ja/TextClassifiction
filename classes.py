import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import textblob.classifiers as cl
import urllib2
import math
from bs4 import BeautifulSoup
from bs4.element import Comment
import cfscrape as cfs

class storydata:
    def __init__(self):
        self.stories = []
        self.ratings = []
        self.urls = []
        self.roundedratings = []
        self.collectStories()
        self.plotRatings()

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
            filename = '/home/jelmer/Documents/pastadata/' + str(i) + '.txt'
            f = open(filename, 'r')
            self.ratings.append(float(f.readlines()[0]))
            #self.stories.append(file.read()[4:])
            f.close()



    def roundRatings(self):
        avg = sum(self.ratings) / float(len(self.ratings))
        for r in self.ratings:
            if (r > avg):
                self.roundedratings.add('good')
            else:
                self.roundedratings.append('bad')

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

    def tag_visible(self,element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def getTrain(self):
        return None

    def getTest(self):
        return None
