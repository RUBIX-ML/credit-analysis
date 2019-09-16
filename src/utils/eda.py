import pandas as pd
import numpy as np
from datetime import datetime
import pandas_profiling
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
import re
from plots import *


def df_summary(df): 
    """
    Print dataframe summary

    Args:
        df: pd.DataFrame, input data

    Returns:
        n_obs: number of observations
        n_features: number of features
        mem_usage: memory usage
        num_cols: numerical features
        cat_cols: categorical features
        date_cols: datetime features
        bool_cols: boolean features
        null_perct: percentage of null cells
    """
    num_cols = feature_types(df)['num_cols']
    cat_cols = feature_types(df)['cat_cols']
    date_cols = feature_types(df)['date_cols']
    bool_cols = feature_types(df)['bool_cols']

    n_obs = df.shape[0]
    n_features = df.shape[1]
    mem_usage = str(round(sum(df.memory_usage())/1024/1024, 2)) + 'MB'
    feature_dtypes = df.dtypes.value_counts().to_dict()
    null_perct = str(sum(df.isnull().sum(axis=0))*100/df.shape[0]) + '%'
    
    print(' Number of rows:', n_obs, '\n', 
          'Number of features:', n_features, '\n', 
          'Memory usage:', mem_usage, '\n\n', 
          'Data types:', '\n',
          'Numerical Columns:', len(num_cols), '\n',
          'Categorical Columns:', len(cat_cols), '\n',
          'Datetime Columns:', len(date_cols), '\n',
          'Boolean Columns:', len(bool_cols), '\n\n',
          'Empty cells:', null_perct)
    return n_obs, n_features, mem_usage, num_cols, cat_cols, date_cols, bool_cols, null_perct


def feature_types(df):
    """
    Get feature types

    Args:
        df: pd.DataFrame, input data
    Returns:
        f_types: feature type
    """
    ftypes = {}
    ftypes['num_cols'] = [str(x) for x in df.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns]
    ftypes['cat_cols'] = [str(x) for x in df.select_dtypes(include=['object', 'category']).columns]
    ftypes['date_cols'] = [str(x) for x in df.select_dtypes(include=['datetime']).columns]
    ftypes['bool_cols'] = [str(x) for x in df.select_dtypes(include=['bool']).columns]
    return ftypes

def feature_stats(df, col_type):
    """
    Calculate feature statistical information

    Args:
        df: pd.DataFrame, input data
        col_type: column/feature type

    Returns:
        max: max value
        min: min value
        median: median value
        std: standard deviation
        missing: number of missing values
        cat: number of categories
        val_counts: value counts of each category 
    """

    num_cols = feature_types(df)['num_cols']
    cat_cols = feature_types(df)['cat_cols']
    date_cols = feature_types(df)['date_cols']
    bool_cols = feature_types(df)['bool_cols']
    
    if col_type == 'numerical':
        for col in num_cols:
            max = df[col].max(axis=0, skipna=True)
            min = df[col].min(axis=0, skipna=True)
            median =df[col].median(axis=0, skipna=True)
            std = df[col].std(axis=0, skipna=True)
            missing = df[col].isnull().sum(axis=0)
        
            print('Feature Name:', col, '\n', 
                  '- max:',  max, '\n',
                  '- min:', min, '\n',
                  '- median:', median, '\n',
                  '- std:', std, '\n',
                  '- missing:', missing)
            histogram_plot(df, col)
        return max, min, median, std, missing
    
    if col_type == 'categorical':
        for col in [*cat_cols, *bool_cols]:
            cats = len(df[col].unique())
            missing = df[col].isnull().sum(axis=0)
            val_counts = df[col].value_counts().to_dict()
            print('Feature Name:', col, '\n', 
                      '- categories:',  cats, '\n',
                      '- missing:', missing, '\n')
            print('Value Counts:')
            for key, value in val_counts.items(): print('-', key, ':', value)
            histogram_plot(df, col)
        return cats, missing, val_counts
    
def convert_cat_codes(df, convert_cols):
    """
    Convert categorical value to one-hot encoding codes

    Args:
        df: pd.DataFrame, input data
        conver_cols: columns/features to be converted
    """

    for col in convert_cols:
        df[col] = df[col].cat.codes
    
def corr_test(df, cols, method, threashold=0):
    """
    Test correlations between features

    Args:
        df: pd.DataFrame, input data
        cols: columns/features
        method: calculation method
        threashold: show columns only has a correlation p > threadhold

    Returns:
        df_corr: Dataframe of the correlation scores
    """

    df_corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    corr_dict = {}
    threashold = threashold
    i = 0
    for x in cols:
        i += 1
        for y in cols[i:]:
            if method == 'pearsonr':
                corr = stats.pearsonr(df[x], df[y])
            if method == 'spearmanr':
                corr = stats.spearmanr(df[x], df[y])
            if method == 'kendalltau':
                corr = stats.kendalltau(df[x], df[y])
            df_corr.loc[x,y] = corr[0]
            df_p.loc[x,y] = corr[1]
            if threashold < corr[0] < 1: 
                key = str(x) + ' is highly correlated with ' + str(y)
                corr_dict[key] = 'p: ' + str(corr[0]) 
    corr_list = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    print(*corr_list, sep = "\n")
    return df_corr


def feature_importance(df, y, model, method):
    """
    Calculate feature importance
    Args:
        df: pd.DataFrame, input data
        y: test data or predictions
        model: the model for testing. e.g. random forest
        method: shuffle or drop feature when calculating scores

    Returns:
        result: dataframe containing importance scores 
    """

    if model == 'RFC':
        m = RandomForestClassifier(n_estimators=10, max_features=1, min_samples_leaf=5, n_jobs=-1)
    if model == 'RFR':
        m = RandomForestRegressir(n_estimators=10, max_features=1, min_samples_leaf=5, n_jobs=-1)
    m.fit(df, y)
    o = m.score(df,y)
    print('original score:', o)
    result = {}
    
    for col in df.columns:
        dff = df.copy()
        if method == 'shuffle':
            dff[col] = np.random.permutation(dff[col])
        if method == 'remove':
            dff = dff.drop(columns=[col])
        m.fit(dff, y)
        d = m.score(dff,y)
        print(col, ': ',  d, ' impact: ', round((o - d)*100/o, 2), '%')
        result[col] = round((o - d)*100/o, 2)
    return result
