import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import spacy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
RandomForestClassifier)
from multiprocessing import Pool



# Spacy Stopwords
en = spacy.load('en_core_web_md')
stopwords = en.Defaults.stop_words


# Model dictionary containing the different elements 
model_dict = {'mnb': # 1st level 1st key
              
              {'model': # 2nd level 1st key
               
               ('mnb', MultinomialNB()),
               
               'params':# 2nd level 2nd key
               
               {'tfv__max_features': [5000,6000],
                'tfv__min_df': [1, 2, 5, 10],
                'tfv__max_df': [0.50, 0.75, 0.9, 0.95, 1.0],
                'tfv__ngram_range': [(1,2)]}},
              
              'bnb': # 1st level 3rd key
              
              {'model': # 2nd level 1st key
               
               ('bnb', BernoulliNB()),
               
               'params':# 2nd level 2nd key
               
               {'tfv__max_features': [5000,6000],
                'tfv__min_df': [1, 2, 5],
                'tfv__max_df': [0.75, 0.9, 0.95, 1.0],
                'tfv__ngram_range': [(1,2)],
              }},
              
              'rfc': # 1st level 2nd key
              
              {'model': # 2nd level 1st key
               
               ('rfc', RandomForestClassifier()),
               
               'params':# 2nd level 2nd key
               
               {'tfv__max_features': [5000,6000],
                'tfv__min_df': [1, 2, 5, 10],
                'tfv__max_df': [0.50, 0.75, 0.9, 0.95, 1.0],
                'tfv__ngram_range': [(1,2)],
                "rfc__n_estimators": [100, 200],
                "rfc__max_depth": [8],
                "rfc__min_samples_leaf": [3, 5, 7],
                "rfc__min_samples_split": [5, 7, 10]}},
              
              'logreg': # 1st level 3rd key
              
              {'model': # 2nd level 1st key
               
               ('logreg', LogisticRegression(max_iter=1000)),
               
               'params':# 2nd level 2nd key
               
               {'tfv__max_features': [5000,6000],
                'tfv__min_df': [1, 2, 5],
                'tfv__max_df': [0.75, 0.9, 0.95, 1.0],
                'tfv__ngram_range': [(1,2)]
              }},
              
              'abc': # 1st level 3rd key
              
              {'model': # 2nd level 1st key
               
               ('abc', AdaBoostClassifier()),
               
               'params':# 2nd level 2nd key
               
               {'tfv__max_features': [5000,6000],
                'tfv__min_df': [1, 2, 5],
                'tfv__max_df': [0.75, 0.9, 0.95, 1.0],
                'tfv__ngram_range': [(1,2)],
                'abc__estimator': [DecisionTreeClassifier(max_depth=2)],
                'abc__learning_rate': np.logspace(-3,0,4),
                'abc__n_estimators': [50, 100, 200]
              }},
              
              'bgc': # 1st level 3rd key
              
              {'model': # 2nd level 1st key
               
               ('bgc', BaggingClassifier()),
               
               'params':# 2nd level 2nd key
               
               {'tfv__max_features': [5000,6000],
                'tfv__min_df': [1, 2, 5],
                'tfv__max_df': [0.75, 0.9, 0.95, 1.0],
                'tfv__ngram_range': [(1,2)],
                'bgc__estimator': [DecisionTreeClassifier(max_depth=2)],
                'bgc__max_samples': np.linspace(0.1, 1.0, 10),
                'bgc__max_features': np.linspace(0.1, 1.0, 10),
                'bgc__n_estimators': [10, 50, 100, 200]
              }}
             
             
             }


def preprocess_for_models_pool(Xdf):
    """
    This function is designed to be used with .apply() on X_train and X_test. When called, the function first sets:
    1. SpaCy Stopwords
    2. nltk.tokenize.word_tokenize as tokenizer
    3. PorterStemmer as stemmer
    Afterwards, it calls tokenizer on the variable passed and saves it as a new temp variable called tokens. Once tokens is set, we use a list comprehension to save a new list of stemmed tokens that are not in spacy stopwords and return the dataframe to be saved as X_train or X_test.
    """
    
    # Set Spacy Stopwords
    en = spacy.load('en_core_web_md')
    stopwords = en.Defaults.stop_words

    # Set Tokenizer using NLTK word_tokenize
    tokenizer = nltk.tokenize.word_tokenize

    # Set PorterStemmer
    stemmer = PorterStemmer()

    # tokens dataframe version
    Xdf['text'] = Xdf['text'].apply(tokenizer)
    
    # Remove tokens that are in Spacy stopwords dataframe version
    Xdf['text'] = Xdf['text'].apply(lambda tokens: [token for token in tokens if token not in stopwords])
    
    # Stem the tokens using PorterStemmer dataframe version
    Xdf['text'] = Xdf['text'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
    
    # return dataframe that has been tokenized, stop words removed, and stemmed.
    return Xdf['text']

def tokenize_pool_processing(Xdfs: dict):
    """
    This function was designed to speed up my tokenize, remove stopwords, and stem process originally coded above as preprocess_for_models. This function takes in 1 parameter which is a dictionary of X_train and X_test.
    
    We iterate through the dictionary, and on each iteration we:
    1. Set the number of processes to use
    2. Set a chunk_size by dividing the length of our X by the number of processes with no decimal
    3. create a list of dataframe chunk slices with a list comprehension:
        a. iterate over the range of indices from 0 to len(X) in steps of chunk size
        b. create a slice of X with rows equal to index and index plus chunk size
        c. creating ~equal slices of X that do not overlap but contain all rows
    4. Create a pool object with the number of processes
    5. Use the pool.map function to pass:
        a. preprocess_for_models_pool function above
        b. chunks to aplly the function too
    6. concatenate the results and add to the dictionary the processed X with the current iteration key.
    7. Return the processed_X dictionary to unpack.
    
    Sources:
    1. https://superfastpython.com/multiprocessing-pool-python/
    2. https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
    3. https://docs.python.org/dev/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
    4. https://stackoverflow.com/questions/41240067/pandas-and-multiprocessing-memory-management-splitting-a-dataframe-into-multipl
    """
    
    processed_X = {}
    for key, X in Xdfs.items():
        # Define the number of processes to use
        num_processes = 4

        # Split the data frame into ~even-sized chunks with no decimal
        chunk_size = len(X) // num_processes
        # set chunks equal to a list comprehension that:
        # 1. iterates over the range of indicies from 0 to length of X_train in steps of 'chunk_size'
        # 2. creates a slice of X_train with rows equal to index and index plus chunk_size in order to
        # create a number of processes that do not overlap but do contain all rows
        chunks = [X[i:i+chunk_size] for i in range(0, len(X), chunk_size)]

        # Create a pool object with the number of processes
        with Pool(num_processes) as pool:
        # Use the pool map function to apply my preprocess for models function to each chunk of the dataframe
        # map takes two parameters: a function to apply, and an iterable with elements to apply the function to
            results = pool.map(preprocess_for_models_pool, chunks)

        # Concatenate the results into a single dataframe
        df_processed = pd.concat(results, axis = 0)
        # Create DataFrame
        processed_X[key] = (df_processed.to_frame(name='text'))

    return processed_X


def get_best_models(X_train, y_train):
    """
    This function takes in only two parameters: X_train and y_train. The function is designed to find the best estimators and params for the different models selected in the model dictionary and return them for scoring. 
    
    The function loops through the model dictionary instantiating a new pipeline per iteration that by default includes CountVectorizer to prepare the tokenized and stemmed data for GridSearchCV. Each iteration appends a different estimator to the pipeline to be passed into GridSearchCV with its own specified hyper params.
    
    After each fit is concluded, it saves a dictionary with the key associated to the model type and the values of: best_estimator_, best_params_, and pipe.
    """
    
    # set dictionary to house best estimators and params
    best_models = {}
    # for loop through the 3 1st level keys of model_model dict
    for key, value in model_dict.items():
        # set pipeline with Count Vectorizer
        pipe = Pipeline([('tfv', TfidfVectorizer())])
        # append current model iteration tuple
        pipe.steps.append(value['model'])
        print(pipe)
        rs = RandomizedSearchCV(pipe,# current pipe iteration
                                value['params'], # current params iteration
                                cv=5, # cross validation 5
                                random_state=42, # random state for consistent results
                                n_jobs=-1) # unlock available CPU for processing
        # fit the current iteration of RandomizedSearchCV
        rs.fit(X_train, y_train)
        # save the best model, best params, and the pipe for scoring 
        best_models[key] = {'model': rs.best_estimator_,
                            'params': rs.best_params_}
    return best_models


def record_scores(baseline, X_train, y_train, X_test, y_test, best_models, model_name:str, df_scores=None):
    """
    This function was originally designed by Devin Fay, instructor for DSIR-221 to capture the scores of multiple model iterations in a dataframe. I have taken out the confusion matrix display and added a couple of different lines, but want to give credit where credit is due.
    
    Changes I made were:
    1. Pass the baseline model for reference
    2. Created a version column and model_type column to keep track of times run
    3. Iteration over a dictionary of best estimators
    """
    if df_scores is None:
        df_scores = pd.DataFrame(columns = ['version','baseline','model_type','train_acc', 'test_acc', 'bal_acc', 'recall', 'precision', 'f1_score'])
    # version equals length of df_scores divided by number of models (6 here)
    version = 'v' + (str((len(df_scores)//6)+1))
        
    # iterate through best_models dictionary
    for model, model_dict in best_models.items():
        # set type
        model_type = model
        # per model, fit the RandomizedSearchCV best_estimator_
        model_dict['model'].fit(X_train, y_train)
        # create predictions
        preds = model_dict['model'].predict(X_test)
        # score accuracy for training data
        train_acc = model_dict['model'].score(X_train, y_train)
        # score accuracy for test data
        test_acc = model_dict['model'].score(X_test, y_test)
        # score balanced accuracy
        bal_acc = balanced_accuracy_score(y_test, preds)
        # calculate recall (TP/(TP+FN))
        rec = recall_score(y_test, preds)
        # calculate precision (TP/(TP+FP))
        prec = precision_score(y_test, preds)
        # calculate F1, ratio of Precision and recall
        fone = f1_score(y_test, preds)
        # create row identifier
        model_name_df = model_name + '_' + model
        # create row for model
        df_scores.loc[model_name_df,:] = [version, baseline, model_type, train_acc, test_acc, bal_acc, rec, prec, fone]

    return df_scores


# Deprecated Functions/Functions no longer in use:

def preprocess_for_models(X):
    """
    DEPRECATED
    Note For the Grader: I kept this function in the model_methods for you to see the base for my preprocess_for_models_pool function. 
    
    This function is designed to be used with .apply() on X_train and X_test. When called, the function first sets:
    1. SpaCy Stopwords
    2. nltk.tokenize.word_tokenize as tokenizer
    3. PorterStemmer as stemmer
    Afterwards, it calls tokenizer on the variable passed and saves it as a new temp variable called tokens. Once tokens is set, we use a list comprehension to save a new list of stemmed tokens that are not in spacy stopwords and return the dataframe to be saved as X_train or X_test.
    """
    
    # Set Spacy Stopwords
    en = spacy.load('en_core_web_md')
    stopwords = en.Defaults.stop_words

    # Set Tokenizer using NLTK word_tokenize
    tokenizer = nltk.tokenize.word_tokenize

    # Set PorterStemmer
    stemmer = PorterStemmer()

    # tokens
    tokens = tokenizer(X)

    # List Comp Tokens if token not in Spacy Stop words
    tokens_keep = [stemmer.stem(token) for token in tokens if token not in stopwords]
    
    # return dataframe that has been tokenized, stop words removed, and stemmed.
    return pd.DataFrame({'stemmed_kept_token': tokens_keep})
