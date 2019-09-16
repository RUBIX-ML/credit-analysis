# CreditAnalysis

Credit Analysis with a group of models Random Forest, Xgboost and LightGBM, etc.
The goal is to predict whether a customer if able to pay the full balance in the next month based on historical payment record.

# Methodology
* Run exploratory analysis such as feature importance to aid feature selection.
* Use grid search to find the best hyper-parameters of different models.
* Automate training and testing process and visualize the results.



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
`pip install -r requirement.txt`

## Packaging
Refer to this link to make an installable python project package

https://packaging.python.org/tutorials/packaging-projects/


# Config files
## Before runing the program:

### Edit your configurations in config/config.yml

**Example:**

**MODEL_DIR:** `'../models'`

**MODEL_LIST:**

  `- 'RandomForest'`
 
  `- 'LightGBM'`
  
  `- 'AdaBoost'`
  
  `- 'ExtraTree'`
  
**DATA_FILE_PATH: **`'../data/raw/credits.csv'`

**REPORT_DIR: **`'../reports'`

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

**After adding configs to config.yml and search_space.yml, run the application by issuing the following command:**

`./run_application`

**The results are located in `Reports` directory**

**Models with best results are located in `models` directory**

## Run tests
 Test cases are located in `test` directory with sample config files
 Run test case with following command:
 `python3 test_models.py` 
 