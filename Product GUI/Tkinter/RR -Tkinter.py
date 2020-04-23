#!/usr/bin/env python
# coding: utf-8

# In[4]:


###################################################################################
# Module Imports

# GUI Modules
from tkinter import *
from tkinter import filedialog
import tkinter.font as tkFont
import os

# Review Scraping Modules
import selenium
from selenium.webdriver import Chrome, ChromeOptions
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
import pandas as pd

# Create Feature Modules
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load("en_core_web_sm")
import re

# Predictor Modules
from sklearn.feature_extraction.text import TfidfVectorizer

#Ranker Module
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# General
import warnings
warnings.filterwarnings('ignore')


####################################################################################
# Function Block

# Get Reviews


def get_review(user_url):
    '''Extracts reviews from user given `flipkart` product page and returns a `pandas dataframe`.

    Parameters
    -----------
    url: Product for which user wants to extract the review
    pages: Number of Pages of reviews the user likes to extract.By default `get_review`
    extracts 25 pages or 250 reviews

    Example
    -------
    >>> df=get_review("https://www.flipkart.com/redmi-8-ruby-red-64-gb/p/itmef9ed5039fca6?pid=MOBFKPYDCVSCZBYR")'''
    global product_name
    pages = 25  # Scrapes 25 Pages By Default
    # User entered url
    url = user_url
    if 'flipkart' in url:
        review_url = url.replace('/p/', '/product-reviews/')

    # Browser Options
    options = Options()
    options.add_argument("--headless")
    options.add_argument('start-maximized')

    # Driver essential to run automated chrome window
    # No option because its in currdir
    driver = webdriver.Chrome(options=options)
    Review_Title, Review_Text, Review_Rating, Upvote, Downvote = [], [], [], [], []

    # Extracting 25 pages of review
    for i in range(1, pages+1):

        # Change web Page
        ping = f'{review_url}&page={i}'
        driver.execute_script('window.open("{}","_self");'.format(ping))

        WebDriverWait(driver, 10).until(EC.staleness_of)

        # Check Read More Buttons
        read_more_btns = driver.find_elements_by_class_name('_1EPkIx')

        # Click on all read more in the current page
        for rm in read_more_btns:
            driver.execute_script("return arguments[0].scrollIntoView();", rm)
            driver.execute_script("window.scrollBy(0, -150);")
            rm.click()

        # Get the product name to save contents inside this folder
        if i == 1:
            product_name = driver.find_element_by_xpath(
                "//div[@class='o9Xx3p _1_odLJ']").text

        # Extracting contents
        # col _390CkK _1gY8H-
        for block in driver.find_elements_by_xpath("//div[@class='col _390CkK _1gY8H-']"):
            Review_Title.append(block.find_element_by_xpath(
                ".//p[@class='_2xg6Ul']").text)
            Review_Text.append(block.find_element_by_xpath(
                ".//div[@class='qwjRop']").text)
            Review_Rating.append(block.find_element_by_xpath(
                ".//div[@class='hGSR34 E_uFuv'or @class='hGSR34 _1x2VEC E_uFuv' or @class='hGSR34 _1nLEql E_uFuv']").text)
            Upvote.append(block.find_element_by_xpath(
                ".//div[@class='_2ZibVB']").text)
            Downvote.append(block.find_element_by_xpath(
                ".//div[@class='_2ZibVB _1FP7V7']").text)

    # Creating df of reviews
    df = pd.DataFrame(data=list(zip(Review_Title, Review_Text, Review_Rating, Upvote, Downvote)), columns=[
                      'Review_Title', 'Review_Text', 'Review_Rating', 'Upvote', 'Downvote'])

    # Handling dtypes of Review_Rating,Upvote,Downvote
    for i in ['Review_Rating', 'Upvote', 'Downvote']:
        df[i] = df[i].astype("int")
    # Return dataframe
    return df

#==================================================================================##
# Create Features

# *******Sub Funtions********

# 1. Sentiment

def sentimental_score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score = vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'

# 2. Target


def target(df):
    df['h'] = np.round(df.Upvote/(df.Upvote+df.Downvote), 2)
    return df

# 3. Drop Unwated Columns


def drop_cols(df):
    drop = ["Sum_of_Up_Down", "Upvote", "Downvote"]
    df = df.drop(drop, axis=1)
    return df

# 4. Number of sentence


def num_sentence(text):
    # return len(nltk.sent_tokenize(text))
    doc = nlp(text)
    return len(list(doc.sents))


# 8. Remove Emoji

def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii').strip()

# 9. Remove Punctuations

def remove_punctuations(text):
    return re.sub('[^\w\s%,-.]', "", text).strip()


#*************************************************************************************#
# *******Main Function*******


def features(df):
    ''' Creates the Feature set based which gave best TEST MAPE during Experimentation
    [Review_Text, Review_Rating,Num_Sentence, h]
    '''
    # Filtering Reviews which has Sum of Upvote and Downvote which is greater than 10
    df['Sum_of_Up_Down'] = df.Upvote-df.Downvote
    df = df[df.Sum_of_Up_Down > 10]

    # Adding New Sentiment Column by calling the function **sentimental_Score**
    df['Sentiment'] = df.Review_Text.apply(sentimental_score)
    # Creating target and dropping unwanted columns
    df = target(df)
    df = drop_cols(df)
    
    # Creating Num_Sentence
    df['Num_Sentence'] = df.Review_Text.apply(num_sentence)


    # Handling Emoji in review_text
    df['Review_Text'] = df.Review_Text.apply(remove_emoji)

    # Remove Punctuations
    df.Review_Text = df.Review_Text.apply(remove_punctuations)

    # Apply Lemmatization for the review and remove stop words
    df.Review_Text = df.Review_Text.apply(lambda text: " ".join(token.lemma_ for token in nlp(text)
                                                                if not token.is_stop))

    return df

#=================================================================================#
# Creates Predictors


def predictor(df, n=0.01):
    '''
    Doc-frequency less than 1 percent will be removed by default..
    Hyperparameter can be changed during function call
    '''
    tfidf = TfidfVectorizer(
        token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b', min_df=n, stop_words="english")
    Matrix = tfidf.fit_transform(df.Review_Text)
    unigram = pd.DataFrame(Matrix.toarray(), columns=tfidf.get_feature_names())
    df=df.select_dtypes(exclude=['object'])
    main = unigram.join(df)
    main = main.fillna(0)
    X = main.drop('h', axis=1)
    y = main.h
    return X, y

#===============================================================================#
# Main Ranker Function

def rank(X,y):
    
    '''
    Rank reviews based on the features created using RandomForestRegressor 
    Which gave the best accuracy during Experimentation
    '''
    #Random Forest Regressor
    rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
    rf.fit(X,y)
    # Predicting on test data
    y_pred = rf.predict(X)
    
    return y_pred

##################################################################################
# GUI BLOCK
root = Tk(baseName="Review Ranker")
root.title("Rank Flipkart Reviews")
root.configure(background='#ffffff')
root.geometry("600x400+400+200")
root.resizable(0, 0)

# Main Title Label
title = Label(root, text="Review Ranker", font="bold 26",
              bg="#00a4fc", padx=183.5, pady=10, borderwidth=0.5, relief="solid").grid(row=0, column=0, sticky="nsew")

# URL Label
url_label = Label(root, text="URL:", font="bold",
                  bg='white', justify="right", bd=1)
url_label.place(height=50, x=100, y=100)

# Folder Label
location_label = Label(root, text="Location:", font="bold",
                       bg='white', justify="right", bd=1)
location_label.place(height=50, x=75, y=200)

# Entry --> String
get_url = Entry(root, width=40, bg="#e6e8e8", borderwidth=1)
get_url.place(width=300, height=30, x=150, y=110)


# Ask folder path
get_folder = Entry(root, width=40, bg="#e6e8e8", borderwidth=1)
get_folder.place(width=300, height=30, x=150, y=210)

# Button --> Browse
folder = StringVar(root)


def browse():
    global folder
    folder = filedialog.askdirectory(initialdir='/')
    get_folder.insert(0, folder)


browse = Button(root, text="Browse", command=browse)
browse.place(height=30, x=475, y=210)


# Button Clear --> Reset all settings to default
def on_clear():
    default_option.set(options[0])
    get_url.delete(0, END)
    get_folder.delete(0, END)


# Function on Submit


def on_submit():
    url = get_url.get()
    old_dir = os.getcwd()
    df = get_review(url)
    df = features(df)
    X, y = predictor(df)
    df['y_pred']=rank(X,y)
    df=df.sort_values(by='y_pred',ascending=False)
    os.chdir(get_folder.get())
    os.makedirs(product_name,exist_ok=True)
    os.chdir(product_name)
    user = df[["Review_Title","Review_Text","Review_Rating","y_pred","Sentiment"]]
    user.to_csv(f"all_ranked_reviews.csv",index=False)
    user[user.Sentiment=='pos'].to_csv("positive.csv",index=False)
    user[user.Sentiment=='neg'].to_csv("negative.csv",index=False)
    os.chdir(old_dir)
    openpath = Button(root, text="Open Folder",
                      command=lambda: os.startfile(get_folder.get()))
    openpath.place(x=380, y=300)


# Button Clear
clear = Button(root, text="Clear", command=on_clear)
clear.place(width=50, x=180, y=300)

# Button -->Submit
submit = Button(root, text="Submit", command=on_submit)
submit.place(width=50, x=280, y=300)

root.mainloop()

