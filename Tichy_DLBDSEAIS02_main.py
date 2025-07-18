from Tichy_DLBDSEAIS02_predef import *
from Tichy_DLBDSEAIS02_ml import *

from sklearn import metrics
import numpy as np
import pandas as pd
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


####################################################
########     Main Control: Start  ########
####################################################
# name of dataset
dataset = './Reviews.csv'
#!!! The Text to analyse must be in a column named "Text".
# !!! The ratings must be in a column named "Score"

#number of rows for analyse
number = 100
# number of text row for an example
no_ex = 5


####################################################
##### Parmeter for predef models  ######
predef_model = 0 # 1 -> on. 0  -> off

#####vader
# 1 => makes an analysis by vaders
do_vader = 0 # 1 -> on. 0  -> off
if do_vader == 1:
# 1 => makes some plots to the vader results 
    plt_vader_res = 1
    
    # compound > pos => marked as 1 compound <  neg marked as -1, between 0.
    vader_pos_limit = 0.3
    vader_neg_limit = -0.3

#####roberta
# 1 => makes example for roberta
do_rob_ex = 0 # 1 -> on. 0  -> off
# 1 => makes an analysis by roberta
do_rob = 0 # 1 -> on. 0  -> off
if do_rob == 1:
    # 1 => makes some plots to the roberta results 
    plt_roberta_res = 1
    # pos = max{pos, neu, neg} -> estimate pos
    # neg = max{pos, neu, neg} -> estimate neg
    # neu = max{pos, neu, neg} -> estimate neu

# 1 cross check. Vader or roberta. show 5 star with no_cc_neg most negative value and 1 star with no_cc_pos most positive.
cc = 1
no_cc_neg = 0
no_cc_pos = 0

####################################################
##### Parmeter for Maschine learning models  ######

ml_model = 1 # 1 -> on. 0  -> off
analyzer = "word" # need for overriding preprocessing

#vetroization 
vect = 2
# 1  # ->bag of words
# 2  # -> TF-IDF

# Preprocessing/Cleaning
lowercase = True

#stopw="None"
stopw="english"

#n grams: 
nmin = 1 # min for ngram
nmax = 1 # max for ngram

#frequency control for words in documents
maxdf=1.0 # ignore terms with frequency higher
mindf= 1.0 # ignore terms with frequency lower

# Selection of method: all four methods can be choosen at the same time. So the f1 score can be compared.
do_lr = 0   # -> Logistic regression if it is 1
do_lda = 1   # -> Linear Discriminat analysis if it is 1
do_dt = 0  # -> Decission Tree if it is 1
do_mb = 1  # -> Multinominal Naive Bayes if it is 1

# For decission tree
crit = "e" # criterion == entropy "e", gini "g"



####################################################
##### First impression and something else  ######

fi = 0 #  1 for a first impression on the dataset. shape..

ba = 0 # 1 for some basic analysis of the dataset

####################################################
########     Main Control: End  ########
####################################################



####################################################
########     Read Data   ########
####################################################

# import the data as csv
df = pd.read_csv(dataset, encoding = 'cp850')
example = df['Text'][no_ex]

####################################################
########     downsize  ########
####################################################

df = df.head(number)

print(df.shape)




####################################################
########     positive - negative - neutral  ########
####################################################

# defines positive, neutral and negative for the voting based on the stars
def create_sentiment(score):
    
    if score==1 or score==2:
        return -1 # negative sentiment
    elif score==4 or score==5:
        return 1 # positive sentiment
    else:
        return 0 # neutral sentiment

df['Sentiment'] = df['Score'].apply(create_sentiment)
  



####################################################
########     first impression of dataset  ########
####################################################

if fi == 1:
    print("Shape of dataset: ", df.shape, "\n")
    
    print("First rows of dataset: ", df.head(), "\n")
    
    print("columns of the dataset: ", df.columns)
    try:
        print("Values for the example: ", df.values[no_ex], "\n")
        
        print("Score for the example: ", df['Score'].values[no_ex], "\n")

    except RuntimeError:
        print("number", no_ex, " not valid for example", "\n")

    

    ### Basic Analyse 
    print(df['Score'].value_counts().sort_index())
    
    ax = df['Score'].value_counts().sort_index().\
        plot(kind='bar', title='Number of reviews for each star', figsize =(10,5))
    
    ax.set_xlabel('Review')
    plt.show()


if ba == 1:
    ### Basic Analyse 
    print(df['Sentiment'].value_counts().sort_index())
    ax = df['Sentiment'].value_counts().sort_index().\
        plot(kind='bar', title='Number Sentiments in reviews', figsize =(10,5))
    
    ax.set_xlabel('Review')
    plt.show()


####################################################
########     call functions  ########
####################################################


# call the machine learn model (more than one is possible)
if ml_model == 1: 
    f, a = ml_analysis(df, vect, lowercase, stopw, nmin, nmax, maxdf, mindf,do_lr, do_lda, do_dt, crit, do_mb, analyzer)
    
# call the predefined model, or the example for roberta
if predef_model == 1: 
    if do_vader == 1:
        f,a = vader_analysis(df, plt_vader_res, vader_pos_limit, vader_neg_limit, cc, no_cc_pos, no_cc_neg)
    if do_rob == 1:
       f,a = roberta_analysis(df, plt_roberta_res, cc, no_cc_pos, no_cc_neg)
    if do_rob_ex == 1: 
        roberta_ex(df, no_ex)




