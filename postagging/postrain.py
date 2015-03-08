#POS tag train

import argparse
import perceplearn
import sys
import re
import codecs
import string

def wordshape(word):
    shape = ""

    if word.isalpha():

        if word.isupper():
            shape = "AA"

        if word[0].isupper():
            shape = "Aa"

        if word.islower():
            shape = "aa"

    elif word.isalnum():
        if word[0].isupper():
            shape = "A0"
        else:
            shape = "a0"

    elif word.isdigit():
        shape = "D0"
    elif len(word)==1 and word in string.punctuation:
        shape = "PUNC"

    return shape

def pos_tag_formatter(input_filename,output_filename, dev_start):

    pos_train = codecs.open(input_filename,"r+",'latin-1',errors = 'ignore')
    pos_feature = codecs.open(output_filename, "w+",'latin-1',errors='ignore')

    line_number = 0
    features = ""
    for line in pos_train:
        line_number += 1

        if dev_start!=0 and line_number>= dev_start:

            pos_feature.write(features)
            pos_feature.close()
            pos_feature = codecs.open("pos.dev.tmp","w+",'latin-1',errors = 'ignore')
            features = ""
            dev_start = 0

        words = re.split(r'\s+',line)
    #First two words are B_O_S, last one words are E_O_S

        for i, comb_word in enumerate(words[2:-2]):
            curr = i+2
            word_tag = comb_word.rpartition('/')
            word = word_tag[0]
            tag = word_tag[2]

            if words[curr-2] == "B_O_S":
                prev2_word =  words[curr-2]
                prev2_tag = ""
            else:
                prev_word_tag = words[curr-2].rpartition('/')
                prev2_word = prev_word_tag[0].lower()
                prev2_tag =  prev_word_tag[2]

            if words[curr-1] == "B_O_S":
                prev1_word = words[curr-1]
                prev1_tag = ""
            else:
                prev_word_tag = words[curr-1].rpartition('/')
                prev1_word = prev_word_tag[0].lower()
                prev1_tag =  prev_word_tag[2]

            if words[curr+1] == "E_O_S":
                next1_word = words[curr+1]
            else:
                next_word_tag = words[curr+1].rpartition('/')
                next1_word = next_word_tag[0].lower()

            if words[curr+2] == "E_O_S":
                next2_word = words[curr+2]
            else:
                next_word_tag = words[curr+2].rpartition('/')
                next2_word = next_word_tag[0].lower()



            features += " ".join([tag , word.lower(),"w1_prev:"+prev1_word,"w1_tag:"+prev1_tag,"w2_prev:"+prev2_word, "w2_tag:"+prev2_tag, "w1_next:"+next1_word,"w2_next:"+next2_word,"w_shape:"+wordshape(word),"\n"])

    pos_feature.write(features)
    pos_feature.close()

    return



def main():

    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("trainingfile",nargs = 1, help = "trainingfile input filename")
    parser.add_argument("modelfile",nargs = 1, help = "modelfile output filename")
    parser.add_argument("-h","--DEVFILE", nargs = 1, help = "optional devfile for error rate calculation")
    args = parser.parse_args()


    #open the training_file and convert to features
    #save it to a file and then pass it to the perceptron

    pos_train = codecs.open(args.trainingfile[0], 'r+',encoding='latin-1',errors = 'ignore')
    train_lines = 0
    i = 0
    for line in pos_train:
        i += 1
    pos_train.close()
    if(args.DEVFILE):
        train_lines = 0
        pos_tag_formatter(args.DEVFILE[0],"pos.dev.fmt",train_lines)
    else:
        train_lines = int(i * .8)


    pos_tag_formatter(args.trainingfile[0],"pos.train.fmt",train_lines)

    #train the perceptron using this formatted file
    if(args.DEVFILE):
        devfile = "pos.dev.fmt"
    else:
        devfile = "pos.dev.tmp"

    perceptron = perceplearn.perceptron_train()
    perceptron.learn("pos.train.fmt",args.modelfile[0],devfile)
    
if __name__ == '__main__':
    main()
