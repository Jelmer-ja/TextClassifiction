import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import textblob.classifiers as cl
import urllib2
from bs4 import BeautifulSoup
import cfscrape as cfs

class storydata:
    def __init__(self):
        self.stories = []
        self.ratings = []
        self.roundedratings = []
        self.urls = []
        self.getUrls()
        self.saveStories()
        self.plotRatings()

    #fill the list of urls with the creepypasta story lins from the text file
    def getUrls(self):
        redflags = ['pastas-indexed-category','discussion-post','wp-content','http:','category-','category/','update','page/']
        blacklist = []
        #Open the file of blacklisted pages, extract the names and close the file again
        blfile = open('blacklist.txt','r')
        for url in blfile.readlines():
            blacklist.append(url[:-2])
        blfile.close()

        #Open the file of website urls, filter out the ones that are not stories themselves
        #Add the rest to the list of urls
        urlfile = open('urls.txt','r')
        for url in urlfile.readlines():
            url2 = url[:-2]
            if(url2 not in blacklist):
                clean = True
                for flag in redflags:
                    if(flag in url2):
                        clean = False
                if(clean):
                    self.urls.append(url2)
        print len(self.urls)

    #Retrieve every story and score for every url in the list and add them to the 'stories' and 'ratings' lists
    def saveStories(self):
        for url in self.urls:
            text,rating = self.getPage(url)
            self.stories.append(text)
            self.ratings.append(rating)
            print len(self.stories)

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

    #Returns the story from the url as plaintext and the rating as a float
    def getPage(self,url):
        #Retrieve the url and create a parser
        scraper = cfs.create_scraper()
        page = scraper.get(url).content
        soup = BeautifulSoup(page, 'html.parser')

        #Find and extract the rating
        rb = soup.find('strong')
        rating = float(rb.text.strip())  # strip() is used to remove starting and trailing

        #Find, extract and merge together the text into a single string
        pees = soup.find_all('p')
        text = ''
        for p in pees:
            if(p.text != 'Explore other pastas:'):
                text += p.text
            else:
                break

        return text, rating

    def getTrain(self):
        return None

    def getTest(self):
        return None
