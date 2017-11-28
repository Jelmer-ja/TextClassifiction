import sklearn as skl
import numpy as np
import scipy as sc
import textblob.classifiers as cl
import urllib2
from bs4 import BeautifulSoup
import cfscrape as cfs
import nltk

class storydata:
    def __init__(self):
        self.stories = []
        self.ratings = []
        self.urls = []
        a, rating = self.getpage('https://www.creepypasta.com/beast-beyond-lake/')

    #fill the list of urls with the creepypasta story lins from the text file
    def getUrls(self):
        pass

    #Retrieve every story and score for every url in the list and add them to the 'stories' and 'ratings' lists
    def saveStories(self):
        pass

    #Returns the story from the url as plaintext and the rating as a float
    def getpage(self,url):
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

        print(text)
        return text, rating


