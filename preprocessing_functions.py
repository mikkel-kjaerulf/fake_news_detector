# DataFrames
import numpy as np
import pandas as pd
from cleantext import clean

# Ploting
import matplotlib.pyplot as plt

# DataCleaning
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

# Other
import time

"""
Tokenizing, stemming, NLTK lib 
"""

def remove_english_stopwords(stopwords):
    def remove_stopwords(tokenlist):
        return filter(lambda x : x not in stopwords, tokenlist)
    return remove_stopwords

def stem_tokens():
    stemmer = PorterStemmer()
    def stem_tokenlist(tokenlist):
        return map(stemmer.stem, tokenlist)
    return stem_tokenlist

def tokenize():
    def tokenize_text(s):
        return list((map(nltk.word_tokenize, s)))
    return tokenize_text

def to_list():
    def turn_to_list(it):
        return list(it)
    return turn_to_list

# Call this function on your dataframe, to pre-process CLEANED data (tokenize, remove stopwords etc.)
def preprocess(dataframe):

    # Tokenize 'content' column
    #print("Tokenizing...")
    start = time.time()
    dataframe['content'] = dataframe['content'].apply(nltk.word_tokenize)
    end = time.time()
    print("Tokenizing took " + str(end - start) + " seconds")

    start = time.time()
    dataframe['content'] = dataframe['content'].apply(remove_english_stopwords(stopwords.words('english')))
    end = time.time()
    print("Removing stopwords took " + str(end - start) + " seconds")

    start = time.time()
    dataframe['content'] = dataframe['content'].apply(stem_tokens())
    end = time.time()
    print("Stemming took " + str(end - start) + " seconds")

    start = time.time()
    dataframe['content'] = dataframe['content'].apply(to_list())
    end = time.time()
    print("Converting to list took " + str(end - start) + " seconds")