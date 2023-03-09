import nltk
import itertools
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import *
import pandas as pd
import regex as re
from cleantext import clean
import pyarrow.feather as feather
from multiprocessing import Pool


pattern = re.compile(r"([\d]{1,2}[\/|\-][\d]{1,2}(?:[\/|\-][\d]{2,4})?|[\d]{2,4}[\/|\-][\d]{1,2}[\/|\-][\d]{1,2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)[\s][\d]{1,2}[a-z][a-z](?:\s[\d]{2,4})|[\d][\d]\w?\w?\sof\s(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s[\d]{2,4})?|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s\d\d?\w?\w?,?(?:\s\d{2,4})?)")
def clean_string(s):
    s1 = pattern.sub("date",s)
    s2 = clean(s1, lower=True,
                no_line_breaks=True,
                no_emails=True,
                no_urls=True,
                no_numbers=True,
                lang="en",
                replace_with_number="num",
                replace_with_email="email",
                replace_with_url="url")
    return s2
def clean_dataframe(dataframe):
    dataframe['content'] = dataframe['content'].apply(clean_string)

def remove_english_stopwords(stopwords):
    def remove_stopwords(tokenlist):
        return np.array(filter(lambda x : x in stopwords, tokenlist))
    return remove_stopwords

def stem_tokens():
    stemmer = PorterStemmer()
    def stem(tokenlist):
        return list(map(stemmer.stem, tokenlist))
    return stem


def preprocess(dataframe):
    # tokenize content column
    print("Tokenizing...")
    dataframe['content'] = dataframe['content'].apply(nltk.word_tokenize)
    tokens = list(itertools.chain.from_iterable(dataframe['content']))
    vocabulary = set(tokens)
    # remove stopwords
    d = stopwords.words('english')
    print("Removing stopwords...")
    dataframe['content'] = dataframe['content'].apply(remove_english_stopwords(d))
    tokens_no_stopwords = list(itertools.chain.from_iterable(dataframe['content']))
    vocabulary_no_stopwords = set(tokens_no_stopwords)
    print("Reduction rate of removing stopwords: " + str(1 - len(vocabulary_no_stopwords) / len(vocabulary)))
    # stem tokens
    print("Stemming...")
    dataframe['content'] = dataframe['content'].apply(stem_tokens())
    tokens_stem = list(itertools.chain.from_iterable(dataframe['content']))
    vocabulary_stem = set(tokens_stem)
    print("Reduction rate of stemming: " + str(1 - len(vocabulary_stem)/len(vocabulary_no_stopwords)))


# TEST PREPROCESSING SPEED
for chunck in pd.read_csv("data/news_cleaned_2018_02_13-1.csv", chunksize=10000):
    print("cleaning...")
    clean_dataframe(chunck)
    print("preprocessing...")
    preprocess(chunck)
    chunck.to_csv("data/news_preprocessed.csv")
    break