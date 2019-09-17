from src.modelling_task import Task
import sys, getopt

def main(argv):
   task = ''
   data = ''
   try:
      opts, args = getopt.getopt(argv,"ht:i:",["task=","data="])
   except getopt.GetoptError:
      print ('Incorrect command! usage: python3 main.py -t <task name> -i <data file name>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('main.py -t <task> -i <data>')
         sys.exit()
      elif opt in ("-t", "--task"):
         task = arg
      elif opt in ("-i", "--data"):
         data = arg
   print ('Task name is:', task, 'Data file is "', data)
   
   task = Task(name=task, data=data)
   task.run_models()
    
if __name__ == "__main__":
    main(sys.argv[1:])
