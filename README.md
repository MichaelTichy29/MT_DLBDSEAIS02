# MT_DLBDSEAIS02
Source code for an Sentiment Analysis project

## What is this done for?
The code is written for the Analysis of the sentiment in reviews. 

## How to handel
The review must be given as csv data. The column with the text must be named "Text". The column with the scores must be named "score". The scores must be given as one to five (stars).

### Tichy_DLBDSEAIS02_main:
In this file there is the "main control". In this the user can choose the algorithm (predefined: Vader or roberta, machine learning: Log. Reg., Decission Tree, Naive Bayes or Linear Discriminanz Analysis)
And differen parameters for the choosen Approach. Further there are options to show some first impression of the dataset, plot results or make a cross check for positive review with 1 star
or negative sentiments with five stars. This method calls the subroutines in _predef or _ml.
Additional the stars are transformed in the labels positive, negative or neutral in this part.

### Tichy_DLBDSEAIS02_main_itter:
Almost the same as in _main. But the options can't be set manual. Instead of this there are loops for parameters and the results are saved in lists which will be transformed to a dataframe and exported to an excel result. 
The input dataset can be changed as well. The requirements for the names of the columns still hold. 
For the professional user the limits of the loops can be changed. This is not set up with parameters in a control mode. 

### Tichy_DLBDSEAIS02_predef:
This is written for the analysis of the text based on the predefined models vader or roberta


### Tichy_DLBDSEAIS02_ml:
This is written for an analysis of the text with different parameters for the dataset cleaning, the vectorization methods the topic extraction method and so on the parameters to use are given in the block main control by the user

## Dataset
The dataset for the analyse can be found here: 
[Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
