#########################################
#
# The following file contains all the
# higher order functions used throughout
# the notebooks, for visual smoothing
# of the notebook order.
#
#########################################


#### Modules ####

# DataFrames
import numpy as np
import pandas as pd

# Ploting
import matplotlib.pyplot as plt

# DataCleaning
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

# Other
import time
import random


#### PRE-PROCESSING FUNCTIONS ####


# RegEx patterns, catches expressions as var-names imply.
mail_pattern = re.compile(r"(?P<Mails>[\w\.-]+@[\w\.-]+\.[\w]+)")
url_pattern = re.compile(r"(?P<URL>(?:(?:https?|ftp):[/][/])?[\w/\-?=%.]+\.[\w/\-&?=%.]+)")
date_pattern = re.compile(r"([\d]{1,2}[\/|\-][\d]{1,2}(?:[\/|\-][\d]{2,4})?|[\d]{2,4}[\/|\-][\d]{1,2}[\/|\-][\d]{1,2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)[\s][\d]{1,2}[a-z][a-z](?:\s[\d]{2,4})|[\d][\d]\w?\w?\sof\s(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s[\d]{2,4})?|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s\d\d?\w?\w?,?(?:\s\d{2,4})?)")
num_pattern = re.compile(r"\w*\d+\w*") 
punct_remove = re.compile(r"[^\w\s]+")
white_space = re.compile(r"[\s]+|\n+")


#Indhenter et sample på ca. 1,45M artikler, meget tidseffektivt
def get_1_point_5_M_sample(csvstring: str):
    random.seed(0)
    n = 11000000 #number of records in file (excludes header)
    s = 1900000 #desired sample size
    skip = sorted(random.sample(range(1,n+1),n-s))
    # Read the CSV file, skipping the randomly selected rows
    sampled_data = pd.read_csv(csvstring, on_bad_lines='skip', skiprows=skip, index_col=0)
    sampled_data = sampled_data.reset_index()
    return sampled_data




"""
Cleans the text from mail/url/date/whitespace/punctuation
"""
def clean_string(st):
    try:
        assert isinstance(st, str)
    except:
        raise ValueError("Input should be a string. Dataframe may already be tokenized.")
    s1 = st.lower()
    s2 = mail_pattern.sub("mail", s1)
    s3 = url_pattern.sub("url", s2)
    s4 = date_pattern.sub("date", s3)
    s5 = num_pattern.sub("num", s4)
    s6 = punct_remove.sub(" ", s5)
    s7 = white_space.sub(" ", s6)
    return s7

# Call this function to clean the dataframe, BEFORE tokenizing.
def clean_dataframe(dataframe):
    start = time.time()
    dataframe['content'] = dataframe['content'].apply(clean_string)
    end = time.time()
    print("cleaning took " + str(end - start) + " seconds")


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
    
    # Failsafe. Raises an error in case we are running twice // try to feed it with the tokenized string rather than a list.
    try:
        assert isinstance(dataframe['content'][0], str)
    except AssertionError:
        raise ValueError("Input should be a string")

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


#### PLOTTING FUNCTIONS #####


def pair_keyword_type(keyword, dataframe): 
    dataarray = []
    for i in range(0, dataframe.__len__()): 
        instances = dataframe.content[i].count(str(keyword))
        dataarray.append((instances, dataframe.type[i]))
    return dataarray


def scatterplot_keyword_type(keyword, dataframe):
    pair_keyword_list = pair_keyword_type(keyword, dataframe)
    tuple_array = pair_keyword_type(keyword, dataframe)
    typearray = []
    countarray = []
    for i in range(0,pair_keyword_list.__len__()): 
        typearray.append(str(tuple_array[i][1]))
        countarray.append(tuple_array[i][0])
    fig = plt.figure(figsize=(10,5))
    ax =fig.add_axes([0,0,2,1])
    ax.scatter(typearray, countarray)
    plt.xticks(rotation = 'vertical')
    plt.show()


def barplot_keyword_type(keyword, dataframe):
    import pandas as pd # Didn't work without this, for some reason.
    tuple_array = pair_keyword_type(keyword, dataframe)
    rawdict = {}
    for i in range(0,len(tuple_array)):
        if tuple_array[i][1] not in rawdict:
            rawdict[tuple_array[i][1]] = [tuple_array[i][0]]
        else:
            rawdict[tuple_array[i][1]].append(tuple_array[i][0])

    meandict={}
    for i in rawdict: 
        meandict[i] = np.mean(rawdict[i])
    mediandict={}
    for i in rawdict: 
        mediandict[i] = np.median(rawdict[i])
    meanvals =[]
    medianvals=[]
    for i in mediandict:
        medianvals.append(mediandict[i])
    for i in meandict:
        meanvals.append(meandict[i])
    types = list(mediandict.keys())
    types = [str(i) for i in types]
    print(medianvals)
    print(meanvals)
    print(types)
    df = pd.DataFrame({
    'types': types,
    'meanvals': meanvals,
    'medianvals': medianvals
    })
    df.plot(x="types", y=["meanvals", "medianvals"], kind="bar", figsize=(10,5))


def boxplot_keyword_type(keyword, dataframe):
    tuple_array=pair_keyword_type(keyword, dataframe) 
    boxdata=[]
    dict = {}
    for i in range(0,len(tuple_array)):
        if tuple_array[i][1] not in dict:
            dict[tuple_array[i][1]] = [tuple_array[i][0]]
        else:
            dict[tuple_array[i][1]].append(tuple_array[i][0])
    for i in dict:
        boxdata.append(dict[i])
    fig = plt.figure(figsize=(10,5))
    ax =fig.add_axes([0,0,2,1])
    ax.boxplot(boxdata, labels = list(dict.keys()))
    plt.xticks(rotation = 'vertical')
    plt.show()