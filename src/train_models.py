import os
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import logging, json, yaml


# %%
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
# %%

with open("../config/config.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)

with open("../config/search_space.yml", 'r') as ymlfile:
    search_space = yaml.load(ymlfile)

MODEL_DIR = config['MODEL_DIR']
MODEL_LIST = config['MODEL_LIST']
DATA_FILE_PATH = config['DATA_FILE_PATH']
REPORT_DIR = config['REPORT_DIR']
FIGURES_DIR = config['FIGURES_DIR']
N_CV_FOLDS = config['N_CV_FOLDS']
TEST_RATIO = config['TEST_RATIO']
COL_LABEL = config['COL_LABEL']
DENSE_FEATURES = config['DENSE_FEATURES']
SPARSE_FEATURES = config['SPARSE_FEATURES']
RF_PARAMS = search_space['RANDOM_FOREST']
EXTRA_TREE = search_space['EXTRA_TREE']
ADA_BOOST = search_space['ADA_BOOST']
GRADIENT_BOOST = search_space['GRADIENT_BOOST']
XGB = search_space['XGB']
LGB = search_space['LGB']


# RANDOM_SEED = 2333
# np.random.seed(RANDOM_SEED)

def proc_data():
    """
    1. Load dataset into pd.DataFrame
    2. Simple feature engineering
        2.1 Convert sparse features to Categorial type
        2.2 Scale dense features to 0 - 1
    3. Feed into different models

    """
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    df = pd.read_csv(DATA_FILE_PATH)

    # Transform categorical features
    if len(SPARSE_FEATURES) > 0:
        for feat in SPARSE_FEATURES:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

        for col in SPARSE_FEATURES:
            df[col] = df[col].astype('category')

    # Transform numerical features
    if len(DENSE_FEATURES) > 0:
        mms = MinMaxScaler(feature_range=(0, 1))
        df[DENSE_FEATURES] = mms.fit_transform(df[DENSE_FEATURES])

    train_set, test_set = train_test_split(df, test_size=TEST_RATIO)

    X_train = train_set[SPARSE_FEATURES + DENSE_FEATURES]
    y_train = train_set[COL_LABEL].values

    logger.info(X_train.head())
    logger.info(X_train.info())

    X_test = test_set[SPARSE_FEATURES + DENSE_FEATURES]
    y_test = test_set[COL_LABEL].values

    return X_train, y_train, X_test, y_test
    

def load_models():
    """ Load the models with configurations

    Returns:
        RandomForest: RF model with params
        ExtraTree: extra_tree model with params
        AdaBoost:  ada_boost model with params
        GradientBoost: gradient boost model with params
        LightGBM: lightGBM model with params
    """
    return {
        'RandomForest': {
            'model_fn': lambda: RandomForestClassifier(),
            'params': RF_PARAMS
        },

        'ExtraTree': {
            'model_fn': lambda: ExtraTreesClassifier(),
            'params': EXTRA_TREE
        },

        'AdaBoost': {
            'model_fn': lambda: AdaBoostClassifier(),
            'params': ADA_BOOST
        },

        'GradientBoost': {
            'model_fn': lambda: GradientBoostingClassifier(),
            'params': GRADIENT_BOOST
        },

        'Xgboost': {
            'model_fn': lambda: XGBClassifier(),
            'params': XGB
        },

        'LightGBM': {
            'model_fn': lambda: LGBMClassifier(),
            'params': LGB
        },
    }


def train_model(model_name, X_train, y_train, X_test, y_test):
    """
    Train the specific model using X_train, y_train, X_test, y_test

    Args:
        model_name: str
        X_train: pd.DataFrame
        y_train: np.array
        X_test: pd.DataFrame
        y_test: np.array

    Returns:
        best_model: best model after grid search
        df_metrics: the best metrics evaluated by X_test and y_test
    """
    logger.info(
        '============= Start Grid CV: ({}) ==================='.format(model_name))


    models = load_models()
    model = models[model_name]['model_fn']()
    param_grid = models[model_name]['params']
    grid_cv = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=N_CV_FOLDS, scoring='roc_auc')
    grid_cv.fit(X_train, y_train)

    best_model = grid_cv.best_estimator_

    logger.info(best_model)
    logger.info('------------------------------------------------------------')
    logger.info('best model score: {}'.format(grid_cv.best_score_))
    logger.info('------------------------------------------------------------')

    logger.info('test set:')
    logger.info('------------------------------------------------------------')
    y_test_pred = best_model.predict(X_test)
    y_test_pred = y_test_pred.reshape(-1)

    y_test_pred_p = best_model.predict_proba(X_test)[:, 1]
    y_test_pred_p = y_test_pred_p.reshape(-1)

    acc_test = accuracy_score(y_test, y_test_pred)
    p_test = precision_score(y_test, y_test_pred)
    r_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_p)
    c_matrix = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = c_matrix.ravel()

    metrics_data = OrderedDict()
    metrics_data['model'] = model_name
    metrics_data['accuracy'] = round(acc_test, 3)
    metrics_data['precision'] = round(p_test, 3)
    metrics_data['recall'] = round(r_test, 3)
    metrics_data['f1_score'] = round(f1_test, 3)
    metrics_data['roc_auc'] = round(roc_auc_test, 3)
    metrics_data['tn'] = tn
    metrics_data['fp'] = fp
    metrics_data['fn'] = fn
    metrics_data['tp'] = tp
    metrics_data['best_params'] = json.dumps(grid_cv.best_params_)

    df_metrics = pd.DataFrame([metrics_data])
    logger.info(df_metrics)

    # feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_data = OrderedDict()
        feature_data['feature'] = X_train.columns
        feature_data['importance'] = best_model.feature_importances_
        df_feature = pd.DataFrame(feature_data)
        df_feature = df_feature.sort_values('importance', ascending=False)
        df_feature.to_csv(f'{REPORT_DIR}/{model_name}_features.csv', index=False)

    df_result_test = pd.DataFrame(
        {'y_true': y_test, 'y_pred': y_test_pred, 'y_pred_p': y_test_pred_p})
    df_result_test.to_csv(
        f'{REPORT_DIR}/{model_name}_result.csv', index=False)

    df_result_test['y_pred_p'].hist(bins=20)
    plt.savefig(f'{FIGURES_DIR}/{model_name}_histogram.png')
    plt.close()

    logger.info(
        '============= End Grid CV: ({}) ===================\n\r\n\r\n\r'.format(model_name))
    return best_model, df_metrics


def run_models():
    all_metrics = []
    for model_name in MODEL_LIST:
        X_train, y_train, X_test, y_test = proc_data()
        best_model, metrics = train_model(model_name, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)

        with open(f'{MODEL_DIR}/{model_name}_best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

    df_all_metrics = pd.concat(all_metrics)
    df_all_metrics.to_csv(f'{REPORT_DIR}/models_metrics.csv', index=False)

    return 0