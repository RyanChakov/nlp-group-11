""" Loads the data from the csv file and returns pandas dataframe that has been preprocessed """
import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')
VALUES = ['ACHIEVEMENT', 'BENEVOLENCE', 'CONFORMITY', 'HEDONISM', 'POWER', 'SECURITY', 'SELF-DIRECTION', 'STIMULATION', 'TRADITION', 'UNIVERSALISM']

def load_full_data():
    """ Loads full dataframe with all values for a statement. Target is 10-dimensional vector with 9 zero values"""
    df = pd.DataFrame()

    for value in VALUES:
        df_value = pd.read_csv(os.path.join(DATA_PATH, value + '.csv'))
        df_value = df_value.rename({'label' : value}, axis = 'columns')
        df_value['label'] = value
        df = pd.concat([df, df_value])

    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    return df

def load_value_data(value):
    assert value in VALUES

    df = pd.read_csv(os.path.join(DATA_PATH, value + '.csv'))
    df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    
    return df

