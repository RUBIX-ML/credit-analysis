import yaml


class Config():
    """Class of configurations
    """

    def __init__(self):
        self.CONFIG = None
        self.SEARCH_SPACE = None

    def load_config(self):
        """Load configuration and search space parameters from testing files
        """
        try: 
            with open("config.yml", 'r') as ymlfile:
                self.CONFIG = yaml.load(ymlfile)

        except IOError:
            print('cannot open config.yml')
        
        try:
            with open("search_space.yml", 'r') as ymlfile:
                self.SEARCH_SPACE = yaml.load(ymlfile)

        except IOError:
            print('cannot open search_space.yml')


    def get_config(self):
        """Check configurations in config and search space files
           Print the configs and seach parameters
        """
        print('==============config.yml==============')
        for key, value in self.CONFIG.items():
            if not key or not value:
                print('Error loading configurations')
            else:
                print(key, '->', value)

        print('==============search_space.yml==============')
        for key, value in self.SEARCH_SPACE.items():
            if not key or not value:
                print('Error loading search_space configurations')
            else:
                print(key, '->', value)

        
def main():
    config = Config()
    config.load_config()
    config.get_config()
    

if __name__ == "__main__":
    main()