#POS tag train

import argparse
import opt_perceplearn
import sys
import re

def main():
    
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("trainingfile",nargs = 1, help = "trainingfile input filename")
    parser.add_argument("modelfile",nargs = 1, help = "modelfile output filename")
    parser.add_argument("-h","--DEVFILE", nargs = "?", help = "optional devfile for error rate calculation")
    args = parser.parse_args()
    
    
    #open the training_file and convert to features
    #save it to a file and then pass it to the perceptron
    
    pos_train = open(args.trainingfile[0], 'r+')
    pos_feature = open("pos.train.fmt",'w+')
    for line in pos_train:
        comb_word_tag = re.split(r'\s+',line.rstrip())
        
        i=0
        #first token processing
        word_tag = re.split(r'/',comb_word_tag[0])
        if len(comb_word_tag) > 1:
            next_tag = re.split(r'/',comb_word_tag[1])
        else:
            next_tag[0] = "E_O_S"
        feature = word_tag[1]+" "+word_tag[0]+ " w_prev:B_O_S"+" w_next:"+next_tag[0]
        pos_feature.write(feature+"\n")
        for token in comb_word_tag[1:-1]:
            feature = ""
            i += 1
            word_tag = re.split(r'/',token)
            prev_tag = re.split(r'/',comb_word_tag[i-1])
            next_tag = re.split(r'/',comb_word_tag[i+1])
            
            feature += word_tag[1]+" "+word_tag[0]+ " w_prev:"+prev_tag[0]+" w_next:"+next_tag[0]
            pos_feature.write(feature+"\n")
        
        #last token processing
        if len(comb_word_tag) > 1:
            prev_tag = re.split(r'/',comb_word_tag[-2])
            word_tag = re.split(r'/',comb_word_tag[-1])
            feature = word_tag[1]+" "+word_tag[0]+" w_prev:"+prev_tag[0]+" w_next:E_O_S"
            pos_feature.write(feature+"\n")
    
    pos_train.close()
    pos_feature.close()
    
    #train the perceptron using this formatted file
    
    perceptron = opt_perceplearn.perceptron_train()
    perceptron.learn("pos.train.fmt",args.modelfile[0],0)
    
if __name__ == '__main__':
    main()
