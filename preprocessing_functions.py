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

def stopwords_remover():
    stop = stopwords.words('english')
    def remove_stopwords(tokenlist):
        return filter(lambda x : x not in stop, tokenlist)
    return remove_stopwords

def token_stemmer():
    stemmer = PorterStemmer()
    def stem_tokenlist(tokenlist):
        return map(stemmer.stem, tokenlist)
    return stem_tokenlist

def tokenizer():
    def tokenize(str):
        return nltk.word_tokenize(str)
    return tokenize

def list_converter():
    def turn_to_list(it):
        return list(it)
    return turn_to_list

# Call this function on your dataframe, to pre-process CLEANED data (tokenize, remove stopwords etc.)
def preprocess(dataframe):

    # Tokenize 'content' column
    #print("Tokenizing...")
    start = time.time()
    dataframe['content'] = dataframe['content'].apply(tokenizer)
    end = time.time()
    print("Tokenizing took " + str(end - start) + " seconds")

    start = time.time()
    dataframe['content'] = dataframe['content'].apply(stopwords_remover(stopwords.words('english')))
    end = time.time()
    print("Removing stopwords took " + str(end - start) + " seconds")

    start = time.time()
    dataframe['content'] = dataframe['content'].apply(stem_tokens())
    end = time.time()
    print("Stemming took " + str(end - start) + " seconds")

    start = time.time()
    dataframe['content'] = dataframe['content'].apply(list_converter())
    end = time.time()
    print("Converting to list took " + str(end - start) + " seconds")