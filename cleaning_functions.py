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

# RegEx patterns, catches expressions as var-names imply.
mail_pattern = re.compile(r"(?P<Mails>[\w\.-]+@[\w\.-]+\.[\w]+)")
url_pattern = re.compile(r"(?P<URL>(?:(?:https?|ftp):[/][/])?[\w/\-?=%.]+\.[\w/\-&?=%.]+)")
date_pattern = re.compile(r"([\d]{1,2}[\/|\-][\d]{1,2}(?:[\/|\-][\d]{2,4})?|[\d]{2,4}[\/|\-][\d]{1,2}[\/|\-][\d]{1,2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)[\s][\d]{1,2}[a-z][a-z](?:\s[\d]{2,4})|[\d][\d]\w?\w?\sof\s(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s[\d]{2,4})?|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s\d\d?\w?\w?,?(?:\s\d{2,4})?)")
num_pattern = re.compile(r"[a-zA-Z]*\d+[a-zA-Z]*") 
punct_remove = re.compile(r"[^a-zA-Z\s]+")
white_space = re.compile(r"[\s]+|\n+")



"""
Cleans the text from mail/url/date/whitespace/punctuation
"""
def clean_single_string(st):
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
    dataframe['content'] = dataframe['content'].apply(clean_single_string)
    end = time.time()
    print("cleaning took " + str(end - start) + " seconds")


#Klassificerer alt som reliable/fake
def binary_labels(df): 
    #klassificerer alle de artikler vi vil bruge ind i reliable eller fake
    df.type = df.type.replace({'political': 'reliable', 'junksci': 'fake', 'bias' : 'fake', 'satire': 'fake', 'conspiracy': 'fake', 'rumor': 'fake', 'unreliable' : 'fake', 'clickbait': 'fake', 'hate': 'fake'})
    return df

#Fjerner alle de artikler vi ikke skal bruges. NB: SKAL kaldes på en dataframe, som allerede har været igennem binary_labels
def remove_bad_articles(df): 
    #fjerner artikler som har volapyk types (inklusiv 'unknown')
    df = df[(df.type == 'reliable') |(df.type == 'fake')]
    #fjerner artikler som ikke har nogen type
    df = df[df.type.notnull()]
    #fjerner artikler uden content
    df = df[df.content.notnull()]
    #fjerner duplerede artikler, ud over en enkelt
    df = df.drop_duplicates(subset = 'content', keep = 'last')
    #fjerner de artikler som ikke indeholder mindst et latinsk bogstav
    df = df[df.content.str.contains('[a-z]')]
    #reset index gør, at hvis vi fjerner artikle [2], bliver artikel [3] rykket ned på index [2] osv. dernedad.
    df = df.reset_index()
    return df 

#Returnerer et dictionary, som repræsenterer den procentmæssige fordeling af typerne i et dataframe
def labelperc(df): 
    labeldict = {}
    for i in df.type: 
        if i in labeldict: 
            labeldict[i] +=1
        else: 
            labeldict[i] = 1
    for i in labeldict: 
        labeldict[i] = labeldict[i]/len(df)*100
    return labeldict
