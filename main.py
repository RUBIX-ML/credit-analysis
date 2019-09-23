from src.task import Task
import sys, getopt, argparse


def main(argv):
   """Receive args from command line and pass them to the task instance 
   """
   # Create the parser
   parser = argparse.ArgumentParser(description='Run the task of grid search')

   # Add the arguments
   parser.add_argument('task',type=str, help=' - name of the task')
   parser.add_argument('file',type=str, help=' - file name of the data file')

   # Execute the parse_args() method
   task_name = parser.parse_args().task
   file_name = parser.parse_args().file

   task = Task(task_name, file_name)
   task.run_models()
    
if __name__ == "__main__":
    main(sys.argv[1:])
