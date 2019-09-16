# CreditAnalysis

Credit Analysis with Random Forest, Xgboost and LightGBM

## File structure
- main.py (main function)
- plots.py (plotly functions for visualization)
- eda.py (exploratory data analysis functions)
- EDA_Report.htm (Feature Engineering process, data wrangling and EDA visualizations)
- Docs/metrics-result.png (Final Scores)
- notebooks (individual jupyter notebooks of RF and Boosting model training and testing)

## Data source
- data.csv
* This application aims at predicting customers’ default credit card payments.
Attribute Information:
* Default payment (Yes = 1, No = 0), as the response variable. 
* X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
* X2: Gender (1 = male; 2 = female). 
* X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
* X4: Marital status (1 = married; 2 = single; 3 = others). 
* X5: Age (year). 
* X6 - X11: History of past payment.
* X6 = the repayment status in September, 2005; 
* X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005
* The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 
* X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; 
* X13 = amount of bill statement in August, 2005; . . .; 
* X17 = amount of bill statement in April, 2005. 
* X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005. 

## Main function uses grid search to tune the hyperparameters of different models. 
### Before runing the program:

1. Edit your configurations in [main.py](main.py)

    ```python
    # output dir of trained models
    MODEL_DIR = 'tmp-model'
    # training data file path
    DATA_FILE_PATH = 'credits.csv'
    # number of cross-validation folds
    N_CV_FOLDS = 5
    # the ratio of test data set
    TEST_RATIO = 0.2

    # the target to be predicted
    COL_LABEL = 'default payment next month'
    # numberical feature names
    DENSE_FEATURES = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
    # categorical feature names
    SPARSE_FEATURES = ['SEX', 'EDUCATION', 'MARRIAGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    # list of models to be trained
    MODEL_LIST = ['extra_tree', 'ada_boost', 'random_forest', 'lgb']
    ```

2. (optional) Edit your grid search hyperparameters for specific models in `load_models()` of [main.py](main.py)
    ``` python
    def load_models():
        return {
            ...,
            'random_forest': {
                'model_fn': lambda: RandomForestClassifier(n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 150],
                    # 'warm_start': True,
                    # # 'max_features': 1.0,
                    # 'max_depth': 6,
                    # 'min_samples_leaf': 2,
                    # 'max_features': 'sqrt',
                }
            },
            ...
        }
    ```
3. Run `python main.py` to train models. It will output the metrics for different models evaluated on the test dataset. Please check `{MODEL_DIR}` for more details.
   ![model results](./docs/metrics-result.png)
