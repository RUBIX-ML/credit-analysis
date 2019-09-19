from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import sys, os, logging, json, yaml
from src.utils.import_models import *


class Task():
    """Class of a task to run the entire process
    Attributes:
        name (str): name of the task.
        CONFIG_PATH (str): path of the config files.
        DATA_PATH (str): path of the data file.
        config(dict): configurations loaded from yml.
        search_space(dict): search parameters for grid search, loaded from yml.
        logger (obj.logging): logging instance.
    """
    def __init__(self, name, data):
        self.name = name
        self.CONFIG_PATH = os.path.join('config', name)
        self.DATA_PATH =os.path.join('data/raw', data)
        self.config = None
        self.search_space = None
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        
    def load_config(self):
        """Processing configuration files
            Read .yml config files
            Assign directories to each task and update configs
        """

        config_file = os.path.join(self.CONFIG_PATH, 'config.yml')
        with open(config_file, 'r') as ymlfile:
            self.config = yaml.load(ymlfile)
            self.config['MODEL_DIR'] = os.path.join(self.config['MODEL_DIR'], self.name)
            self.config['REPORT_DIR'] = os.path.join(self.config['REPORT_DIR'], self.name)
            self.config['FIGURES_DIR'] = os.path.join(self.config['FIGURES_DIR'],self.name)

        search_space_file = os.path.join(self.CONFIG_PATH, 'search_space.yml')
        with open(search_space_file, 'r') as ymlfile:
            self.search_space = yaml.load(ymlfile)


    def feature_transform(self):
        """Processing dataset for model training
        1. Load dataset into pd.DataFrame
        2. Convert sparse features to Categorial type

        Returns:
            df(pd.Dataframe): dataframe with transformed feature types
        """

    # Transform categorical features
        if not os.path.exists(self.DATA_PATH):
            sys.exit('File not found!')

        df = pd.read_csv(self.DATA_PATH)
        
        if len(self.config['SPARSE_FEATURES']) > 0:
            for feat in self.config['SPARSE_FEATURES']:
                lbe = LabelEncoder()
                df[feat] = lbe.fit_transform(df[feat])

            for col in self.config['SPARSE_FEATURES']:
                df[col] = df[col].astype('category')

        # Transform numerical features
        if len(self.config['DENSE_FEATURES']) > 0:
            mms = MinMaxScaler(feature_range=(0, 1))
            df[self.config['DENSE_FEATURES']] = mms.fit_transform(df[self.config['DENSE_FEATURES']])
        
        return df
    
    
    def split_datasets(self):
        """Processing dataset for model training
            Splite datasets for training and testing

        Returns:
            X_train: predictors of training data
            y_train: predictions of training data
            X_test:  predictors of test data
            y_test:  predictions of test data
        """

        df = self.feature_transform()
        train_set, test_set = train_test_split(df, test_size=self.config['TEST_RATIO'])
        X_train = train_set[self.config['SPARSE_FEATURES'] + self.config['DENSE_FEATURES']]
        y_train = train_set[self.config['COL_LABEL']].values

        self.logger.info(X_train.head())
        self.logger.info(X_train.info())

        X_test = test_set[self.config['SPARSE_FEATURES'] + self.config['DENSE_FEATURES']]
        y_test = test_set[self.config['COL_LABEL']].values

        return X_train, y_train, X_test, y_test


    def load_model(self, model_name):
        """Train the specific model using X_train, y_train, X_test, y_test

        Args:
            model_name: name of the model

        Returns:
            model: model function
            param_grid: the parameter grid for grid search
        """

        model =  eval(self.config['MODEL_LIST'][model_name]['model_fn'])
        param_grid = self.search_space[model_name]
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
        
        self.logger.info('============= Start Grid CV: ({}) ==================='.format(model_name))
        
        X_train, y_train, X_test, y_test = self.split_datasets()
        model, param_grid = self.load_model(model_name)

        for dir in [self.config['MODEL_DIR'], self.config['REPORT_DIR'], self.config['FIGURES_DIR']]:
            if not os.path.exists(dir): 
                os.mkdir(dir)

        grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=self.config['N_CV_FOLDS'], scoring='roc_auc')
        grid_cv.fit(X_train, y_train)
        best_model = grid_cv.best_estimator_
        self.get_feature_importance(model_name, X_train, best_model)
        self.logger.info(best_model)
        self.logger.info('------------------------------------------------------------')
        self.logger.info('best model score: {}'.format(grid_cv.best_score_))
        self.logger.info('------------------------------------------------------------')

        self.logger.info('test set:')
        self.logger.info('------------------------------------------------------------')
        
        return model_name, grid_cv, best_model, X_test, y_test


    def get_feature_importance(self, model_name, X_train, best_model):
        """Generate feature importance report for best model

        Args:
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
                df_feature.to_csv(f'{self.config["REPORT_DIR"]}/{model_name}_features.csv', index=False)
    

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
        self.logger.info(df_metrics)

        df_result_test = pd.DataFrame(
            {'y_true': y_test, 'y_pred': y_test_pred, 'y_pred_p': y_test_pred_p})
        df_result_test.to_csv(
            f'{self.config["REPORT_DIR"]}/{model_name}_result.csv', index=False)

        df_result_test['y_pred_p'].hist(bins=20)
        plt.savefig(f'{self.config["FIGURES_DIR"]}/{model_name}_histogram.png')
        plt.close()

        self.logger.info(
        '============= End Grid CV: ({}) ===================\n\r\n\r\n\r'.format(model_name))
        return df_metrics


    def run_models(self):
        """run all the models defined in confi file
        Save model in pkl format
        Save scores in csv files
        """
        self.load_config()
        all_metrics = []

        for model_name in self.config['RUN_MODELS']:
            model_name, grid_cv, best_model, X_test, y_test = self.train_model(model_name)
            metrics = self.gen_metrics(model_name, grid_cv, best_model, X_test, y_test)
            all_metrics.append(metrics)
            

            with open(f'{self.config["MODEL_DIR"]}/{model_name}_best_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)

        df_all_metrics = pd.concat(all_metrics)
        df_all_metrics.to_csv(f'{self.config["REPORT_DIR"]}/models_metrics.csv', index=False)