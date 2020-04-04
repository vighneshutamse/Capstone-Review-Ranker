#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing Necessary Packages
import os
import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import filedialog, messagebox
import spacy
from spacy.lang.en import English
from spacy.pipeline import SentenceSegmenter
nlp = spacy.load('en_core_web_sm')
# VaderSentiment is used for splitting reviews into positive and negative sentiments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#Ignoring Errors
import warnings
warnings.filterwarnings('ignore')
import emoji
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from xgboost import XGBRegressor

#Creating a window in tkinter making it of fixed width and non-resizeable
window = tk.Tk()
window.geometry("600x400")
window.resizable(0,0)
window.title("Rank Reviews")


#Declaring Tkinter Variables
entryfile=tk.StringVar()
folderpath=tk.StringVar()

#Select File Name Function
def selectfilename():
    global filename
    filename=tk.filedialog.askopenfilename(initialdir=os.getcwd())
    print(filename)
    entryfile.set(filename)

# #Location to save Function
# def selectfolder():
#     global foldername
#     foldername=tk.filedialog.askdirectory(initialdir=os.getcwd())
#     print(foldername)
#     folderpath.set(foldername)

#Submit Button Function Command
def submit():
    data=filename
    print(data)
#     directory=foldername
#     print(directory)
    rank(data,save=None)
    done.config(text="Reviews Ranked and Stored in specified directory")

#All Sub function of Rank --> To create features

# VADER sentiment analysis tool for getting pos, neg and neu.
# VADER (Valence Aware Dictionary and sEntiment Reasoner)
def sentimental_Score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'

# Create Target
def target(df):
    df['h']=np.round(df.Upvote/(df.Upvote+df.Downvote),2)
    return df

#Dropping Unwanted Columns
def drop_cols(df):
    drop=["Sum_of_Up_Down","Upvote","Downvote"]
    df=df.drop(drop,axis=1)
    return df

# Number of Sentence
def num_sentence(text):
    #return len(nltk.sent_tokenize(text))
    doc = nlp(text)
    return len(list(doc.sents))

# Number of Upper case words (Fully Upper)
def count_upper(text):
    count=0
    for i in text.split():
        if text.isupper():
            count+=1
    return count

# Number of words with Proper Format
def count_proper(text):
    count=0
    for i in text.split():
        if text.istitle():
            count+=1
    return count

#Number of Emoji
def emoji_count(text):
    return emoji.emoji_count(text)

#Remove emoji from review Text
def remove_emoji(text):
    return text.encode('ascii','ignore').decode('ascii').strip()

#Remove Punctuations
def remove_punctuations(text):
    return re.sub('[^\w\s%,-.]',"",text).strip()

#Add POS tag for each word
def pos_tag(text):
    doc=nlp(text)
    return ' '.join([token.pos_ for token in doc])

#Percentage of Nouns
def Noun(text):
    text_len=len(text.split())
    noun_count=0
    for word in text.split():
        if word=='NOUN':
            noun_count+=1
    return np.round((noun_count/text_len)*100,2)

#Percentage of Verb
def Verb(text):
    text_len=len(text.split())
    verb_count=0
    for word in text.split():
        if word=='VERB':
            verb_count+=1
    return np.round((verb_count/text_len)*100,2)

#Percentage of Adverb
def Adverb(text):
    text_len=len(text.split())
    adv_count=0
    for word in text.split():
        if word=='ADV':
            adv_count+=1
    return np.round((adv_count/text_len)*100,2)

#Percentage of Adjective
def Adj(text):
    text_len=len(text.split())
    adj_count=0
    for word in text.split():
        if word=='ADJ':
            adj_count+=1
    return np.round((adj_count/text_len)*100,2)

#Creates features for the current df
def features(df):    
    #Filtering Reviews which has Sum of Upvote and Downvote which is greater than 10
    df['Sum_of_Up_Down']=df.Upvote-df.Downvote
    df=df[df.Sum_of_Up_Down>10]

    #Adding New Sentiment Column by calling the function **sentimental_Score**
    df['Sentiment'] = df.Review_Text.apply(sentimental_Score)
    #Creating target and dropping unwanted columns 
    df=target(df)
    df=drop_cols(df)
    
    #Length Before
    df["Len_before"] = df.Review_Text.apply(lambda x: len(x.split()))

    #Creating Num_Sentence
    df['Num_Sentence']=df.Review_Text.apply(num_sentence)

    #Number of Question Mark
    df['No_QMark'] = df.Review_Text.str.count(pat='\?')

    #Number of Exclamatio Mark
    df['No_ExMark']=df.Review_Text.str.count(pat='!')

    #Number of Upper Case Text
    df['No_Upper']=df.Review_Text.apply(count_upper)

    #Number of Proper Case Text
    df['No_proper']=df.Review_Text.apply(count_proper)

    #Count of Emoji
    df['Emoji_Count']=df.Review_Text.apply(emoji_count)

    #Handling Emoji in review_text
    df['Review_Text']=df.Review_Text.apply(remove_emoji)

    #Remove Punctuations
    df.Review_Text=df.Review_Text.apply(remove_punctuations)

    #Removed spell correction because its taking time in TextBlob

    #Apply Lemmatization for the review and remove stop words
    df.Review_Text=df.Review_Text.apply(lambda text: " ".join(token.lemma_ for token in nlp(text) 
                                               if not token.is_stop)) 

    #Length of the Review After removing stop words
    df["Len_after"] = df.Review_Text.apply(lambda x: len(x.split()))

    #Applying POS for all words
    df['POS']=df.Review_Text.apply(pos_tag)

    #To avoid Zero Division Error
    df=df[df.Len_after>=1]

    #Percentage of Noun
    df['Perc_Noun']=df.POS.apply(Noun)

    #Percentage of Verb
    df['Perc_Verb']=df.POS.apply(Verb)

    #Percentage of Adverb
    df['Perc_Adverb']=df.POS.apply(Adverb)

    #Percentage of Adjective
    df['Perc_Adj']=df.POS.apply(Adj)
    
    return df
def predictor(df,n=1):
    '''
    Pass the df for which important features with tfidf is needed
    n represents the percentage of document in which it should occur or else its neglected as important feature
    '''
    count=CountVectorizer(token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b')
    count_matrix=count.fit_transform(df.Review_Text)
    bow=pd.DataFrame(count_matrix.toarray(),columns=count.get_feature_names())
    bow_sum=pd.DataFrame(bow.sum(axis=0),columns=['sum_count'])

    #getting the column names of words which occured more than 1 of the times in the entire corpus
    important = list(bow_sum[bow_sum.sum_count>len(df)*n/100].index)

    tfidf= TfidfVectorizer(token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b')
    Matrix=tfidf.fit_transform(df.Review_Text)
    unigram=pd.DataFrame(Matrix.toarray(),columns=tfidf.get_feature_names())
    unigram=unigram[important]
    df=df.drop(['Review_Title','Review_Text','POS','Sentiment'],axis=1)
    main=unigram.join(df)
    main=main.fillna(0)
    X=main.drop('h',axis=1)
    y=main.h
    return X,y
    
#Main Ranking Function
def rank(data,save=None):
    '''
    rank function takes in csv file with `review title`, `review text`, `review rating`,`upvote`
    and `downvote` as an input and ranks the review based on the important unigrams in the corpus and 
    other features and stores it in the user specified path
    '''

    # Setting directory to current directory if No value is provided
    if save == None:
        save = os.getcwd()

    # Get the file type and filename from data
    filename = data[:data.find('.')]
    print(filename)
    filetype = data[data.find('.')+1:]
    print(filetype)
    phone = data[data.rfind('/')+1:data.find('.')]
    # Make directory in the filename and change cwd to that folder to store all data in that
    # Handling Removing of empty directory in the name of filename
    if filename not in os.listdir(save):
        os.mkdir(save+'\\'+phone)
    else:
        os.removedirs(save+'\\'+phone)
        os.mkdir(save+'\\'+phone)

    # Combining file types to call relevant read function
    

    # Evaluating the file based on the filename provided and storing it in the dataframe
    # Handling multiple file types
    try:
        if filetype=='xlsx' or filetype == 'xls':
            df = pd.read_excel(filename+'.'+filetype)
        else:
            read = "pd.read_"+filetype
            #print(data)
            df = eval(read)(data)
    except AttributeError as ae:
        print(ae)
        print(entryfile)
        df = pd.read_csv(filename+'.'+filetype)
    
    # Changing directory to save all graph
    os.chdir(save+'\\'+phone)
    
    #Add in the code here
    df = features(df)
    X,y = predictor(df)
    
    xgb = XGBRegressor(n_estimators = 1000,n_jobs=-1,random_state = 0)
    xgb.fit(X, y)
    df['h_pred']=xgb.predict(X)
    df=df.sort_values(by='h_pred',ascending=False)    
    df.to_csv("allreviews.csv")
    df[df.Sentiment=='pos'].to_csv("positive.csv")
    df[df.Sentiment=='neg'].to_csv("negative.csv")
    
    # Changing directory to parent
    os.chdir('..')

#TITLE FRAME
top_frame = tk.Frame(master=window, height=60, background="blue",bd=1,relief="raised")
top_frame.pack(fill='x',padx=5,pady=5)

#Top Label
top_label = tk.Label(master=top_frame,text="Review Ranking System",font=("Times",30),bg="blue",fg="white").pack()

#MIDDLE FRAME
mid_frame = tk.Frame(master=window,height = 320)
mid_frame.pack(fill="both",padx=5,pady=5)

#Mid Labels

#file and path of file
file = tk.Label(master=mid_frame,text="File: ",font=("Times",15)).place(x=135,y=50)

#Select File Button:
# select = tk.Button(master = mid_frame, text="Select", command=select)
select = filedialog.Button(master = mid_frame, text="Select", command=selectfilename)
select.place(x=520,y=50)

#Entry button
file_entry = tk.Entry(master=mid_frame,textvariable=entryfile,width=50).place(x=200,y=54)

#Location to save the graph
# location=tk.Label(master=mid_frame,text="Location to save: ",font=("Times",15)).place(x=40,y=150)

#Select Location Button
# select2 = filedialog.Button(master = mid_frame, text="Select", command=selectfolder).place(x=520,y=150)

#Folder to save in...
# file_entry = tk.Entry(master=mid_frame,textvariable=folderpath,width=50).place(x=200,y=154)

label = tk.Label(text="Note: ", font=('Times', 11, 'bold')).place(x=130,y=180)
label1 = tk.Label(text="The file should contain the following columns in the specified order:\n`Review_Title`, `Review_Text`, `Review_Rating`,`Upvote` and `Downvote`.",
                 font=('Times', 10)).place(x=130,y=200)


#Submit Button
submit = tk.Button(master=mid_frame,text="Submit",command=submit).place(x=300,y=250)

#result
done = tk.Label(master=mid_frame,text="")
done.place(x=200,y=280)
    
tk.mainloop()

