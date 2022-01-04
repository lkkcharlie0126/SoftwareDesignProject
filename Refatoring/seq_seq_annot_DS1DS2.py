import random

# Self definied function
from functions.run_program import DS1DS2Runner

random.seed(654)
def main():
    programRunner = DS1DS2Runner()
    programRunner.run()

if __name__ == '__main__':
    main()