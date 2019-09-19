# CreditAnalysis

Credit Analysis with a group of models Random Forest, Xgboost and LightGBM, etc.
The goal is to predict whether a customer if able to pay the full balance in the next month based on historical payment record.

# Methodology
##### Automate ML
-  Run exploratory analysis such as feature importance to aid feature selection.
-  Use grid search to find the best hyper-parameters of different models.
-  Automate training and testing process and visualize the results.

##### Simple to Use
- Simply use the config template to create tasks then run the job to get the best results
- Provide interim analysis results. e.g. Feature engineering, AUC score, histograms

##### Open/Close Principle - Easy to expand
- Model training and reporting functions/code are closed to modification in future model expansion process
- Adding new models only requires modifications in config files (config.yml)



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

# Install & Verify
## Install dependencies with pip
`pip install -r requirements.txt`

## Packaging
Refer to this link to make an installable python project package

https://packaging.python.org/tutorials/packaging-projects/



# Adding Configurations
### Before runing the program:

### Edit your configurations in config/config.yml

**Example:**

**MODEL_DIR:** `'../models'`

**MODEL_LIST:**

  `- 'RandomForest'`

  `- 'LightGBM'`

  `- 'AdaBoost'`

  `- 'ExtraTree'`

**DATA_FILE_PATH:** `'../data/raw/credits.csv'`

**REPORT_DIR:** `'../reports'`

**FIGURES_DIR:** `'../reports/figures'`

**N_CV_FOLDS:** `5`

**TEST_RATIO:** `0.2`

**COL_LABEL:** `'default payment next month'`

**DENSE_FEATURES:**

  `- 'LIMIT_BAL'`

 `- 'AGE'`

**SPARSE_FEATURES:**

  `- 'SEX'`

 ` - 'EDUCATION'`

 `- 'MARRIAGE'`



### Edit your grid search configs in config/space_search.yml

**Example:**

**RANDOM_FOREST:**

   **max_features:**

   `- 0.6`

   `- 1.0`

   **max_depth:**

   `- 6`

   `- 10`

   **n_estimators:**

   `- 50`

   `- 100`

# Run program
**For each task, please create a folder with task name under `config` directory (e.g. `config/credit`) then put `config.yml` and `search_space.yml` in this directory.**

**Config templates are available in `config/config_templates` directory**


**Put the data file under `data/raw/` directory.**


**After adding configs to config.yml and search_space.yml, run the application by issuing the following command:**


`python3 main.py -t <task name> -i <data file name>`


Example command:

`python3 main.py -t credit -i credits.csv`




**The results are located in `Reports/<task name>`  and  `Reports/figures/<task name>/` directories**


![model results](./reports/figures/sample_metrics-result.png)

**Models with best results are located in `models` directory**

## Adding new Models
After installing new models and algorithms, just add the new model in `import.py` and `config.yml` to load the model in the tasks.

- Add model package in `import_models.py`.
  * Example: `from scklearn import <model_name>`
- Add new models in the config files - `config.yml: MODEL_LIST`
- Define what models to run by the task in `config.yml: RUN_MODELS`

## Run tests
 Test cases are located in `test` directory with sample config files
 Run test case with following command:

 `python3 test_models.py`
