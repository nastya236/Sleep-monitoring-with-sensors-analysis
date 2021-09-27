import pandas as pd
import numpy as np

def get_labels(path):
    '''
    path: global path to csv file with sleeping stages
    '''
    stages = pd.read_csv(path)
    labels = stages[['Wonambi v6.17']].iloc[:, 0].values[1:]
    set_labels = set(labels)
    set_labels = np.sorted(list(set_labels))
    dict_labels = dict(zip(list(set_labels), np.arange(len(set_labels))))
    result_labels = np.array([dict_labels[i] for i in labels])

    return result_labels


def get_time_for_labels(path):
    '''
    path: global path to csv file with sleeping stages
    '''
    stages = pd.read_csv(path)
    time_array = stages[['Wonambi v6.17']].iloc[:, 0].index[1:]
    time_labels = [i[0] for i in time_array]

    return time_labels