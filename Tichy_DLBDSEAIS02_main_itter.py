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
########     Main Control  ########
####################################################

# For this dataset there is a set of sentiment analysis with predefined models (vader and roberta)
# and some machine learning approaches (Logistic regression, decission tree, Naive Bayes and Linear Discriminant Analysis)
# the loops for the options are predefined. So the only choice ist the dataset and the number of rows of the 
# dataset which should be considered.

# name of dataset
dataset = './Reviews.csv'
#!!! The Text to analyse must be in a column named "Text".
# !!! The ratings must be in a column named "Score"

#number of rows for analyse
number = 1000

####################################################
########     decoding of meth for ML Models  ########
####################################################

# transform the loop variable meth in the on/off structure which machine learn model should be considered
def meth_calc(meth):
    if meth == 0:
        do_lr = 1 # logistic regression
        do_lda = 0 # Linear Discriminanz Analysis
        do_dt = 0 # Decission tree
        do_mb = 0 # Naive Bayes
        n_meth = "log Reg"
    elif meth == 1:
        do_lr = 0
        do_lda = 1
        do_dt = 0
        do_mb = 0
        n_meth = "Linear Disk Analysis"
    elif meth == 2:
        do_lr = 0
        do_lda = 0
        do_dt = 1
        do_mb = 0
        n_meth = "Decission Tree"
    else:
        do_lr = 0
        do_lda = 0
        do_dt = 0
        do_mb = 1
        n_meth = "Naive Bayes"

    return n_meth, do_lr, do_lda, do_dt, do_mb
            



# lists for the parameters

list_method = []
list_crit = []
list_limitpos = []
list_limitneg = []
list_vect = []
list_lower = []
list_nmin = []
list_nmax = []
list_mindf = []
list_maxdf = []
list_f1 = []
list_acc = []
list_cf = []


#################################################
###########  loops for the parameters    #############
#################################################

def Auswertung(dataset):
    

    # import the data as csv
    df = pd.read_csv(dataset, encoding = 'cp850')
    df = df.head(number)

    # defines positive, neutral and negative for the voting based on the stars
    def create_sentiment(score):
        
        if score==1 or score==2:
            return -1 # negative sentiment
        elif score==4 or score==5:
            return 1 # positive sentiment
        else:
            return 0 # neutral sentiment

    df['Sentiment'] = df['Score'].apply(create_sentiment)
    
    # results for the vader method. 
    # The parameter differs for the classification of the output. The compound sentiment is considered. So
    # the text is said to be positive if and only if compound value > vader_pos_limit. 
    # negative, if and only if the compund value < vader_neg_limit
    for pos in range(5):
        for neg in range(5):
            vader_pos_limit = pos * 0.1
            vader_neg_limit = neg* 0.1
            # call of the function 
            f, a, cf = vader_analysis(df, 0, vader_pos_limit, vader_neg_limit, 0, 0, 0)
            
            #appending of parameters and results
            list_method.append("vader")
            list_crit.append("none")
            list_limitpos.append(vader_pos_limit)
            list_limitneg.append(vader_neg_limit)
            list_vect.append("none")
            list_lower.append("none")
            list_nmin.append("none")
            list_nmax.append("none")
            list_mindf.append("none")
            list_maxdf.append("none")
            list_f1.append(f)
            list_acc.append(a)
            list_cf.append(cf)
                
    # Roberta method. -> No variation
    f,a, cf = roberta_analysis(df, 0,0,0,0)
    
    #appending of parameters and results
    list_method.append("Roberta")
    list_crit.append("none")
    list_limitpos.append("none")
    list_limitneg.append("none")
    list_vect.append("none")
    list_lower.append("none")
    list_nmin.append("none")
    list_nmax.append("none")
    list_mindf.append("none")
    list_maxdf.append("none")
    list_f1.append(f)
    list_acc.append(a)
    list_cf.append(cf)
   
    
    
    # ML methods
    mindf = 0.0 #no variation in this parameter
    maxdf = 1.0 #no variation in this parameter
    analyzer = "word"
    for meth in range(4): # (4)# loop for the different methods
        n_meth, do_lr, do_lda, do_dt, do_mb = meth_calc(meth)
        for vect in range(1,3):  # (1,3) # loop for the different vectorization methods
            if vect == 1:
                nvect = "BoW"
            else:
                nvect = "TF-IDF"
            nmin = 1
            for nmax in range(nmin,4): # (nmin,4) # loop for the maximum size of the ngrams. the minimum size is set as 1.
                for lw in range(2):  # (2) #loop if the vectorizer should lowercase the words first.
                    if lw == 0:
                        lowercase = True
                    else:
                        lowercase = False
                        
                    for stop in range(2): # (2) # loop if stopwords should be used. 
                        if stop == 0:
                             stopw = "english"
                        else:
                            stopw = None  
                        if meth != 2:
                            
                            crit = "egal"
                            # call of method
                            try:
                                f,a, cf = ml_analysis(df, vect, lowercase, stopw, nmin, nmax, maxdf, mindf,do_lr, do_lda, do_dt, crit, do_mb, analyzer)
                            except RuntimeError:
                                f = "fail"
                                a = "fail"
                            #appending of parameters and results
                            list_method.append(n_meth)
                            list_crit.append("none")
                            list_limitpos.append("none")
                            list_limitneg.append("none")
                            list_vect.append(nvect)
                            list_lower.append(lowercase)
                            list_nmin.append(nmin)
                            list_nmax.append(nmax)
                            list_mindf.append(mindf)
                            list_maxdf.append(maxdf)
                            list_f1.append(f)
                            list_acc.append(a)
                            list_cf.append(cf)
                           
                        else: 
                            # parameter for the split. Only for the decission tree.
                            for k_crit in range (2): # (2)
                                if k_crit == 0: 
                                    crit = "e"
                                    n_crit = "entropie"
                                else:
                                    crit = "g"
                                    n_crit = "gini"
                                
                                try:
                                    # call of method
                                    f,a, cf = ml_analysis(df, vect, lowercase, stopw, nmin, nmax, maxdf, mindf,do_lr, do_lda, do_dt, crit, do_mb, analyzer)
                                except RuntimeError:
                                    f = "fail"
                                    a = "fail"
                                #appending of parameters and results
                                list_method.append(n_meth)
                                list_crit.append(n_crit)
                                list_limitpos.append("none")
                                list_limitneg.append("none")
                                list_vect.append(nvect)
                                list_lower.append(lowercase)
                                list_nmin.append(nmin)
                                list_nmax.append(nmax)
                                list_mindf.append(mindf)
                                list_maxdf.append(maxdf)
                                list_f1.append(f)
                                list_acc.append(a)
                                list_cf.append(cf)
                               
                                
#call function                            
Auswertung = Auswertung(dataset)

# list to dataframe
dfres = pd.DataFrame([list_method, list_crit, list_limitpos, list_limitneg, list_vect, list_lower, list_nmin, list_nmax, list_mindf, list_maxdf, list_f1, list_acc, list_cf],
                      index=["method", "criterion", "limit pos", "limit neg", "vectorization", "lower", "n gram min", "n gram max", "frequenz min", "frequenz max", "f1 score", "accuracy", "Confusion"])
dfres = dfres.transpose()

# writing Dataframe to Excel
datatoexcel = pd.ExcelWriter('result_Sent.xlsx')

# write DataFrame to excel
dfres.to_excel(datatoexcel)
# save the excel
datatoexcel.close()












