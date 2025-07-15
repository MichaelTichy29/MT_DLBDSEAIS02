import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.feature_extraction.text import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore")



####################################################
########     Clean Data  ########
####################################################


def clean_data(review):
    
    no_punc = re.sub(r'[^\w\s]', '', review)
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])
    
    return(no_digits)


###
# further cleanings? -> not done at the moment
####




############################################
###### Function for results an measure   ##################
############################################
#calculate the f1 score and the accuracy

def results(y_test, preds, y, method):
    
    print("\n")
    print("Results for ", method, "\n")
    # Confusion matrix
    conf = metrics.confusion_matrix(y_test, preds)
    print(conf, "\n")
    # f1 score
    f1 = metrics.f1_score(y_test, preds, average='weighted')
    # avarage =  micro, macro, weighted
    print("f1 = ", f1, "\n")
    # accuracy
    accuracy = metrics.accuracy_score(y_test, preds)
    
    print("accuracy = ", accuracy,  "\n")
    
    if method != "vader" and method != "roberta":
        y_name = np.unique(y)
        print("y_name = ", y_name, "\n")
        frequency_counts = y.value_counts()
        print("Distribution of sentiments in train and test ", frequency_counts)
        
    return f1, accuracy




def ml_analysis(df, vect, lc, stopw, nmin, nmax, maxdf, mindf, do_lr, do_lda, do_dt, crit, do_mb, analyzer):
    df['Text'] = df['Text'].apply(clean_data)

    ####################################################
    ########     Vectorizer  ########
    ####################################################
    # vectorize and tokenize the corpus
    
    if vect == 1: 
        # Count Vectorizer
        vectorizer = CountVectorizer(analyzer=analyzer, lowercase=lc, stop_words=stopw, ngram_range=(nmin, nmax), max_df=maxdf, min_df = mindf)
        X = vectorizer.fit_transform(df['Text'])

    if vect == 2:
        # tfidf Vectorizer
        tfidf = TfidfVectorizer(strip_accents=None, lowercase=lc, preprocessor=None)
        X = tfidf.fit_transform(df['Text'])

    ####################################################
    ########     target, Test split  ########
    ####################################################


    y = df['Sentiment'] # target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)
        
    
    ############################################
    ###### Method     #############################
    ############################################
    
    
    ###### Logistic Reg     #############################

    if do_lr == 1: 
        method = "Log Reg"
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_train,y_train) # fit the model

        preds = lr.predict(X_test) # make predictions

        #calculate the f1 score and the accuracy
        f,a = results(y_test, preds, y, method)


    ###### LDA     #############################
    

    if do_lda == 1:
        method = "LDA"
        
        LDA = LinearDiscriminantAnalysis()
        X_train = X_train.toarray()

        data_projected = LDA.fit_transform(X_train,y_train)

        preds = LDA.predict(X_test)
        
        #calculate the f1 score and the accuracy
        f,a = results(y_test, preds, y, method)

    ###### Decission Tree   ####################
    

    if do_dt == 1: 
        method = "dt"
        if crit == "g": # choose of split criterion. Gini or entropy.
            decision_tree = DecisionTreeClassifier(criterion="gini", max_depth = 100)
        elif crit == "e":
            decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=100)

        decision_tree = decision_tree.fit(X_train, y_train)

        preds = decision_tree.predict(X_test)

        #calculate the f1 score and the accuracy
        f,a = results(y_test, preds, y, method)


    ###### Multinominal Bayes   ####################
    

    if do_mb == 1: 
        method = "mNB"
        # Train a Multinomial Naive Bayes classifier
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        
        # Evaluate the model
        preds = clf.predict(X_test)
        
        #calculate the f1 score and the accuracy
        f,a = results(y_test, preds, y, method)

    return f, a






