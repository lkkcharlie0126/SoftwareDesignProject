import random

# Self definied function
from functions.run_program import AamiRunner

random.seed(654)
def main():
    programRunner = AamiRunner()
    programRunner.run()

if __name__ == '__main__':
    main()