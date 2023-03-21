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
num_pattern = re.compile(r"\w*\d+\w*") 
punct_remove = re.compile(r"[^\w\s]+")
white_space = re.compile(r"[\s]+|\n+")


"""
Cleans the text from mail/url/date/whitespace/punctuation
"""
def clean_single_string(st):
    s1 = clean(st.lower(), lang='en')
    s2 = mail_pattern.sub("mail", s1)
    s3 = url_pattern.sub("url", s2)
    s4 = date_pattern.sub("date", s3)
    s5 = num_pattern.sub("num", s4)
    s6 = clean(s5, no_punct=True)
    s7 = white_space.sub(" ", s6)
    return s7

# Call this function to clean the dataframe, BEFORE tokenizing.
def clean_dataframe(dataframe):
    start = time.time()
    dataframe['content'] = dataframe['content'].apply(clean_single_string)
    end = time.time()
    print("cleaning took " + str(end - start) + " seconds")

