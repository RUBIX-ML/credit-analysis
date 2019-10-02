import sys, os
CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = (os.path.join(CWD, '..'))
sys.path.append(BASE_DIR)

from src.task import *


def main():
    """Use testing config and data file to create a task instance
        Run task and train the models
    """

    print("**************Starting Unit Test!***************")
    
    try:
        task = Task('unit_test', 'test_data.csv')
        task.run_models()
        print("**************Unit test Passed!**************")

    except:
        print('Cannot complete the test case, test failed')

if __name__ == "__main__":
    main()
    