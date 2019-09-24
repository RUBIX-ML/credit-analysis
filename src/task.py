from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import sys, os, logging, json, yaml
from src.classifier import Classifier


class Task():
    """Class of a task to run the entire process
    Attributes:
        task_name (str): name of the task,
        file_name (str): name of the input data file,
        CONFIG_PATH (str): path of the config files,
        DATA_PATH (str): path of the data file,
        CONFIG(dict): configurations loaded from yml,
        SEARCH_SPACE(dict): search parameters for grid search, loaded from yml.
    """
    def __init__(self, task_name, file_name):
        self.task_name = task_name
        self.CONFIG_PATH = os.path.join('config', task_name)
        self.DATA_PATH =os.path.join('data/raw', file_name)
        self.CONFIG = None
        self.SEARCH_SPACE = None

        
    def load_config(self):
        """Processing configuration files
            Read .yml config files,
            Assign directories to each task and update configs.
        """

        config_file = os.path.join(self.CONFIG_PATH, 'config.yml')
        with open(config_file, 'r') as ymlfile:
            self.CONFIG = yaml.load(ymlfile)
            self.CONFIG['MODEL_DIR'] = os.path.join(self.CONFIG['MODEL_DIR'], self.task_name)
            self.CONFIG['REPORT_DIR'] = os.path.join(self.CONFIG['REPORT_DIR'], self.task_name)
            self.CONFIG['FIGURES_DIR'] = os.path.join(self.CONFIG['FIGURES_DIR'],self.task_name)

        search_space_file = os.path.join(self.CONFIG_PATH, 'search_space.yml')
        with open(search_space_file, 'r') as ymlfile:
            self.SEARCH_SPACE = yaml.load(ymlfile)

        return self.CONFIG

    def load_data(self):
        """Read data file
            DATA_PATH is passed from task.config
        """

        if not os.path.exists(self.DATA_PATH):
            sys.exit('File not found!')

        df = pd.read_csv(self.DATA_PATH, low_memory = False)    
        return df


    def run_models(self):
        """run all the models defined in confi file
            Save model in pkl format,
            Save scores in csv files.
        
        Params:
            CONFIG (dict): configurations of task,
            SEARCH_SPACE (dict): hyperparameters of models,
            df (pandas dataframe): data frame read from raw data,
            model: Classifier instance,
            all_metrics (list): list to save all model training results.
        """

        CONFIG = self.load_config()
        DATA_PATH = self.DATA_PATH
        SEARCH_SPACE = self.SEARCH_SPACE
        df = self.load_data()
        model = Classifier(CONFIG, SEARCH_SPACE, df)
        all_metrics = []

        for model_name in self.CONFIG['RUN_MODELS']:
            model_name, grid_cv, best_model, X_test, y_test = model.train_model(model_name)
            metrics = model.gen_metrics(model_name, grid_cv, best_model, X_test, y_test)
            all_metrics.append(metrics)
            

            with open(f'{self.CONFIG["MODEL_DIR"]}/{model_name}_best_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)

        df_all_metrics = pd.concat(all_metrics)
        df_all_metrics.to_csv(f'{self.CONFIG["REPORT_DIR"]}/models_metrics.csv', index=False)