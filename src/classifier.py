from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
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
import sys, os, logging, json, yaml
from src.utils.feature_engineering import feature_transform, split_datasets, gen_feature_importance

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class Classifier():
    """Class of a Classifier
    """
    def __init__(self, CONFIG, SEARCH_SPACE, data):
        self.CONFIG = CONFIG
        self.SEARCH_SPACE = SEARCH_SPACE
        self.data = data
    

    def proc_data(self):
        """Read data file
            DATA_PATH is passed from task.config
        """

        # if not os.path.exists(self.DATA_PATH):
        #     sys.exit('File not found!')

        #df_raw = pd.read_csv(self.DATA_PATH)
        df = feature_transform(self.data, self.CONFIG)
        X_train, y_train, X_test, y_test = split_datasets(df, self.CONFIG)

        return X_train, y_train, X_test, y_test


    def load_model(self, model_name):
        """Train the specific model using X_train, y_train, X_test, y_test

        Args:
            model_name: name of the model

        Returns:
            model: model function
            param_grid: the parameter grid for grid search
        """

        model =  eval(self.CONFIG['MODEL_LIST'][model_name]['model_fn'])
        param_grid = self.SEARCH_SPACE[model_name]
        return model, param_grid
        

    def train_model(self, model_name):
        """Train the specific model using X_train, y_train, X_test, y_test

        Args:
            model_name: name of the model

        Returns:
            model_name: name of the model
            grid_cv: grid cross validation result
            best_model: best model after grid search_space
            X_test:  predictors of test data
            y_test:  predictions of test data
        """
        
        logger.info('============= Start Grid CV: ({}) ==================='.format(model_name))
        
        X_train, y_train, X_test, y_test = self.proc_data()
        model, param_grid = self.load_model(model_name)

        for dir in [self.CONFIG['MODEL_DIR'], self.CONFIG['REPORT_DIR'], self.CONFIG['FIGURES_DIR']]:
            if not os.path.exists(dir): 
                os.mkdir(dir)

        grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=self.CONFIG['N_CV_FOLDS'], scoring='roc_auc')
        grid_cv.fit(X_train, y_train)
        best_model = grid_cv.best_estimator_
        gen_feature_importance(self.CONFIG, model_name, X_train, best_model)
        logger.info(best_model)
        logger.info('------------------------------------------------------------')
        logger.info('best model score: {}'.format(grid_cv.best_score_))
        logger.info('------------------------------------------------------------')

        logger.info('test set:')
        logger.info('------------------------------------------------------------')
        
        return model_name, grid_cv, best_model, X_test, y_test
    

    def gen_metrics(self, model_name, grid_cv, best_model, X_test, y_test):
        """Generate metrics of model training and create reports

        Args:
            model_name: name of the model
            grid_cv: grid cross validation result
            best_model: best model after grid search_space
            X_test:  predictors of test data
            y_test:  predictions of test data

        Returns:
            df_metrics: all metrics in a dataframe
        """

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

        df_result_test = pd.DataFrame(
            {'y_true': y_test, 'y_pred': y_test_pred, 'y_pred_p': y_test_pred_p})
        df_result_test.to_csv(
            f'{self.CONFIG["REPORT_DIR"]}/{model_name}_result.csv', index=False)

        df_result_test['y_pred_p'].hist(bins=20)
        plt.savefig(f'{self.CONFIG["FIGURES_DIR"]}/{model_name}_histogram.png')
        plt.close()

        logger.info(
        '============= End Grid CV: ({}) ===================\n\r\n\r\n\r'.format(model_name))
        return df_metrics