#POS tag train

import argparse
import perceplearn
import sys

def main():
    
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("trainingfile",nargs = "+", help = "trainingfile input filename")
    parser.add_argument("modelfile",nargs = "+", help = "modelfile output filename")
    parser.add_argument("-h", nargs = "?", help = "optional devfile for error rate calculation")
    args = parser.parse_args()
    print(args)
    
    #open the training_file and convert to features
    #save it to a file and 
    


if __name__ == '__main__':
    main()
