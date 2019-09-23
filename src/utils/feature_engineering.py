from collections import OrderedDict
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import pandas as pd


def feature_transform(df, CONFIG):
    """Processing dataset for model training
    1. Load dataset into pd.DataFrame
    2. Convert sparse features to Categorial type

    Returns:
        df(pd.Dataframe): dataframe with transformed feature types
    """

    # Transform categorical features
    if len(CONFIG['SPARSE_FEATURES']) > 0:
        for feat in CONFIG['SPARSE_FEATURES']:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

        for col in CONFIG['SPARSE_FEATURES']:
            df[col] = df[col].astype('category')

    # Transform numerical features
    if len(CONFIG['DENSE_FEATURES']) > 0:
        mms = MinMaxScaler(feature_range=(0, 1))
        df[CONFIG['DENSE_FEATURES']] = mms.fit_transform(df[CONFIG['DENSE_FEATURES']])
        
        return df
    
    
def split_datasets(df, CONFIG):
    """Processing dataset for model training
        Splite datasets for training and testing

        Args:
            df (pandas dataframe): data frame read from raw data
            CONFIG (dict): configurations of task

        Returns:
            X_train: predictors of training data
            y_train: predictions of training data
            X_test:  predictors of test data
            y_test:  predictions of test data
    """

    train_set, test_set = train_test_split(df, test_size=CONFIG['TEST_RATIO'])
    X_train = train_set[CONFIG['SPARSE_FEATURES'] + CONFIG['DENSE_FEATURES']]
    y_train = train_set[CONFIG['COL_LABEL']].values
    X_test = test_set[CONFIG['SPARSE_FEATURES'] + CONFIG['DENSE_FEATURES']]
    y_test = test_set[CONFIG['COL_LABEL']].values

    return X_train, y_train, X_test, y_test


def gen_feature_importance(CONFIG, model_name, X_train, best_model):
    """Generate feature importance report for best model

    Args:
        CONFIG (dict): configurations for model training task 
        model_name: name of the model
        best_model: model with the highest scores
        X_train: predictors of training data
    """
    if hasattr(best_model, 'feature_importances_'):
            feature_data = OrderedDict()
            feature_data['feature'] = X_train.columns
            feature_data['importance'] = best_model.feature_importances_
            df_feature = pd.DataFrame(feature_data)
            df_feature = df_feature.sort_values('importance', ascending=False)
            df_feature.to_csv(f'{CONFIG["REPORT_DIR"]}/{model_name}_features.csv', index=False)