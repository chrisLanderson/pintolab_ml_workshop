# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:44:04 2021

@author: gincley.b
"""
#%% Imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
import glob
from sklearn import metrics

#%% Functions
def importData(root_directory, filetype, rowrange = [], verbose = False):
    '''
    Import the csv and pickled annotation dataframes, then collect them in a list.
    WARNING: This will read files in the sequential order that they appear in the folder.
    Inputs: 
        - root_directory (str) 
        - filetype (str) type of data file e.g. '.csv' or '.pickle'
        - rowrange (list) start and end-1 rows to filter by e.g. [0,500] for 0-499 (remember pythonic number indexing, not excel)
        - verbose (bool) turn True if would like to see file list output
    '''
    print("WARNING: This will read files in the sequential order that they appear in the folder. Enable verbose to confirm.")
    df_list = []
    # Feature .csv
    if filetype == '.csv':
        file_list = list(glob.glob(root_directory+'*.csv'))
        n_files = len(file_list)
        print(f"{n_files} '.csv' files found in this directory.")
        for file in file_list:
            if rowrange:
                df_list.append(pd.read_csv(file)[rowrange[0]:rowrange[1]])
            else:
                df_list.append(pd.read_csv(file))
    # Annotation pickled dataframes
    elif filetype == '.pickle':
        file_list = list(glob.glob(root_directory+'*.pickle'))
        n_files = len(file_list)
        print(f"{n_files} '.pickles' files found in this directory.")
        for file in file_list:
            df_list.append(pickle.load(open(file,'rb')))
    # Unrecognized filetype provided
    else:
        print("Warning: Unrecognized filetype encountered. Please try '.csv' or '.pickle'.")

    if verbose:
        print("Loading files in this order: ")
        for i in range(n_files):
            print(f"{file_list[i][-40:]}")
            
    return df_list


def mergeDFs(f_list,a_list):
    '''
    Merges feature and annotation dataframes. Returns in aggregated list.
    Inputs: 
        - f_list (list) features list of dataframes
        - a_list (list) annotations list of dataframes
    '''
    f_annotated = []
    for n, (f, a) in enumerate(zip(f_list,a_list)):
        add = a["celltype"]
        f_annotated.append(f.join(add))
    return f_annotated


def concatenateDFs(df_list):
    ''' Concatenates dataframes that are grouped in a list
    Inputs:
        - df_list (list) list of pandas dataframes
    '''
# Concatenate from list:
    df = df_list[0]
    for i in range(1,len(df_list)):
        df = pd.concat([df,df_list[i]])
    df = df.reset_index(drop=True)
    return df


def filterByFreq(df: pd.DataFrame, column: str, min_freq: int) -> pd.DataFrame:
    """Filters the DataFrame based on the value frequency in the specified column.
    Inputs:
    :param df: DataFrame to be filtered.
    :param column: Column name that should be frequency filtered.
    :param min_freq: Minimal value frequency for the row to be accepted.
    :return: Frequency filtered DataFrame.
    """
    # Frequencies of each value in the column.
    freq = df[column].value_counts()
    # Select frequent values. Value is in the index.
    frequent_values = freq[freq >= min_freq].index
    # Return only rows with value frequency above threshold.
    return df[df[column].isin(frequent_values)]


def reClass(labels,reclassdict):
    '''Re-classes targets/labels according to a dictionary
    Inputs:
        - labels (ndarray or Series) to be reclassified
        * Must be an array or Series to work, not a list *
        - reclassdict (dictionary):
            - Keys = the class label you want to change to, 
            - Values = the class label to be reassigned
        '''
    if type(labels) is not (np.ndarray or pandas.core.series.Series):
        print("Note: Labels should be of type: ndarray OR Series")
        print("Type passed: ",type(labels))
    for key, value in reclassdict.items():
        print(f"{key} from {value}")
        for val in value:
            mask = [l==val for l in labels]
            labels[mask] = key
    return labels


def saveModel(filename,model,mdltype):
    '''Saves model according to filename
    Inputs:
        - filename (str) path including name of file e.g. ../models/model.mdl
        - model (model) binary object to be saved
        -mdltype (str) type of model e.g. SVM, RF, CNN, etc.
    '''
    if mdltype=='SVM' or mdltype=='RF':
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    elif mdltype=='CNN':
        model.save(filename)
    else:
        print("Unrecognized mdltype.")
        

def confusionMatrixFxn(ground_truth,prediction,label_list,title_add = [],cm=plt.cm.Blues):
    '''Generates a figure for the confusion matrix of a given prediction set, based on scikit-learn 'metrics'
    Inputs:
        - ground_truth (array) of true labels (e.g. yTest)
        - prediction (array) of model predictions
        - label_list (list) of unique class names. Should be in same order as numerically encoded class numbers.
        - title_add (str) optional addition to the title to specify sample, etc.
        - cm (plt.cm.XXXX) pyplot colormap (default is 'Blues')
    '''
    ## Initial parameters
    classes=label_list
    n = len(classes)
    normalized_CM = np.zeros((n,n))
    ##  Normalize confusion matrix
    confusionMatrix = metrics.confusion_matrix(ground_truth,prediction)
    true_totals = np.sum(confusionMatrix,1)
    for i in range(len(true_totals)):
        normalized_CM[:,i] = confusionMatrix[:,i]/true_totals
    ## Generate Figure
    fig, ax = plt.subplots()
    im = ax.imshow(normalized_CM, interpolation='nearest', cmap=cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusionMatrix.shape[1]),
           yticks=np.arange(confusionMatrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Normalized confusion matrix: {title_add}',
           ylabel='True label',
           xlabel='Predicted label')
    ## Configure text appearance and color
    fmt = '.2f'
    for i in range(normalized_CM.shape[0]):
        for j in range(normalized_CM.shape[1]):
            ax.text(j, i, format(normalized_CM[i, j], fmt),
                    ha="center", va="center",
                    color="white" if normalized_CM[i, j] > 0.5 else "black")
    fig.tight_layout()
    #return fig