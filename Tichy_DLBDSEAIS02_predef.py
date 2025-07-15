import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('vader_lexicon')
  
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import seaborn as sns
# for vadder
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
# for roberta
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from Tichy_DLBDSEAIS02_ml import *

import warnings
warnings.filterwarnings("ignore")




####################################################
########    Lexicon Approach: Vader  ########
####################################################


def vader_analysis(df, plt_vader_res, vader_pos_limit, vader_neg_limit, cc, no_cc_pos, no_cc_neg): 
    sia =  SentimentIntensityAnalyzer()
    
    # Example
    #print(sia.polarity_scores("This is a good day"))
    #print(sia.polarity_scores("worst things ever"))
    #print(sia.polarity_scores(example))
    
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text']
        myid = row['Id']
        res[myid] = sia.polarity_scores(text)
    
    vaders = pd.DataFrame(res)
    vaders = vaders.T
    
    vaders = vaders.reset_index().rename(columns={'index':'Id'})
    
    vaders = vaders.merge(df, how='left')
    
    #print(vaders)
    
    ##### Plot the vader results
    if plt_vader_res == 1:
        ax = sns.barplot(data=vaders, x='Score', y = 'compound')
        ax.set_title('Compound score by number of star')
        plt.show()
        
        fix, axs = plt.subplots(1,3, figsize=(15,5))
        sns.barplot(data=vaders, x='Score', y = 'pos', ax = axs[0])
        sns.barplot(data=vaders, x='Score', y = 'neu', ax = axs[1])
        sns.barplot(data=vaders, x='Score', y = 'neg', ax = axs[2])
        axs[0].set_title('Positive')
        axs[1].set_title('Neutral')
        axs[2].set_title('Negative')
        plt.show()
    
    
    # interpret the compound value as positive, negative or neutral, based on the pos/neg_vader_limit.: 
    # compound > 0,3 as positive
    # -0,3 < compound < 0,3 as neutral
    # compound < - 0,3 as negative
    
    def estimate_sentiment(comp):
        
        if comp >= vader_pos_limit:
            return 1 # pos sentiment estimatet by text
        elif comp > vader_neg_limit:
            return 0 # neutral sentiment estimatet by text
        else:
            return -1 # negative sentiment estimatet by text
    
    vaders['Sentiment_est'] = vaders['compound'].apply(estimate_sentiment)
    
    #print(vaders.columns)
    
    ###confusion_matrix
    y_test = vaders['Sentiment'].to_numpy()
    preds = vaders['Sentiment_est'].to_numpy()
    
    #calculate the f1 score and the accuracy
    f,a = results(y_test, preds, y_test, "vader")
    
    return f,a # the results are only for the itter method, to write the results in lists.
    

    
    ####################################################
    ########    check the text to the value  ########
    ####################################################
    
    # check the no_cc_pos most positive (no_cc_neg most negative) value with only one (five) star. Question is weather the estimatet sentiment of the 
    # text fits to one star - in our opinion.
    if cc == 1:
        # Biggest positive value with one score in vaders
        test = vaders.query('Score == 1')\
            .sort_values('pos', ascending=False)['Text'].values[no_cc_pos]
        
        print("Testfall pos, one star ", no_cc_pos, "most positive value: \n" , test, "\n")
        
        # negative sentiment 5 star review
        test = vaders.query('Score == 5')\
            .sort_values('neg', ascending=False)['Text'].values[no_cc_neg]
        
        print("Testfall neg, 5 star ", no_cc_neg, "most negative value: \n ", test, "\n")

####################################################
########    pretrained Approach: Roberta  ########
####################################################


####################################################
######## Roberta Example

# do the roberta model just for one line: no_ex

def roberta_ex(df, no_ex):
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    example = df['Text'][no_ex]
    encoded_text = tokenizer(example, return_tensors='pt')
    print(example)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)
    #negative, neutral, positive
    
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]}
    print(scores_dict)
    


####################################################
######## Roberta method


def roberta_analysis(df, plt_roberta_res, cc, no_cc_pos, no_cc_neg):
    # calculate the roberta sentiment for the hole dataset
    
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)


    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        #print(scores)
        # negative, neutral, positive
        
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2]}
        #print(scores_dict)
        return scores_dict
        
    # try exept is done as there are some comments to large. 
    roberta_result = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Text']
            myid = row['Id']
            roberta_result[myid] = polarity_scores_roberta(text)
            #res[myid] = polarity_scores_roberta(example)
            #print(roberta_result)
        except RuntimeError:
            print(f'Broke for id {myid}')
    
    roberta = pd.DataFrame(roberta_result)
    
    #Transpose:
    roberta = roberta.T
    #print("df roberta, transpose", roberta)
    
    
    roberta = roberta.reset_index().rename(columns={'index':'Id'})
    roberta = roberta.merge(df, how='left')
    
    
    ##### Plot the roberta results
   
    if plt_roberta_res == 1:
        fix, axs = plt.subplots(1,3, figsize=(15,5))
        sns.barplot(data=roberta, x='Score', y = 'roberta_pos', ax = axs[0])
        sns.barplot(data=roberta, x='Score', y = 'roberta_neu', ax = axs[1])
        sns.barplot(data=roberta, x='Score', y = 'roberta_neg', ax = axs[2])
        axs[0].set_title('Positive')
        axs[1].set_title('Neutral')
        axs[2].set_title('Negative')
        plt.show()
    
    
    
    
    # interpret the values to the main sentiment positive, neutral or negative : 
    # pos > max {neu, neg} - > pos
    # neu > max {pos, neg} - > neu
    # neg > max {neu, pos} - > neg
    
    roberta['Sentiment_est'] = 5
    roberta['Sentiment_est'] = np.where((roberta['roberta_pos'] >= roberta['roberta_neu']) & (roberta['roberta_pos'] >= roberta['roberta_neu']), 1, roberta['Sentiment_est']) 
    roberta['Sentiment_est'] = np.where((roberta['roberta_neg'] >= roberta['roberta_neu']) & (roberta['roberta_neg'] >= roberta['roberta_pos']), -1, roberta['Sentiment_est']) 
    roberta['Sentiment_est'] = np.where(roberta['Sentiment_est'] == 5, 0 , roberta['Sentiment_est']) 
    
     
    
    
    ###confusion_matrix
    y_test = roberta['Sentiment'].to_numpy()
    preds = roberta['Sentiment_est'].to_numpy()
    
    
    #calculate the f1 score and the accuracy
    f,a = results(y_test, preds, y_test, "roberta") 
    
    return f,a # the results are only for the itter method, to write the results in lists.

    ####################################################
    ########    check the text to the value  ########
    ####################################################
    # check the no_cc_pos most positive (no_cc_neg most negative) value with only one (five) star. Question is weather the estimatet sentiment of the 
    # text fits to one star - in our opinion.
    
    if cc == 1:
        # Biggest positive value with one score in vaders
        test = roberta.query('Score == 1')\
            .sort_values('roberta_pos', ascending=False)['Text'].values[no_cc_pos]
        
        print("Testfall pos, one star, ", no_cc_pos, "most positive value: \n ", test, "\n")
        
        # negative sentiment 5 star review
        test = roberta.query('Score == 5')\
            .sort_values('roberta_neg', ascending=False)['Text'].values[no_cc_neg]
        
        print("Testfall neg, 5 star = ", no_cc_neg, "most negative value: \n", test, "\n")

    
    
    
    
  




