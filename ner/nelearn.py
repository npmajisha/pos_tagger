__author__ = 'majisha'

#Named Entity Recognition

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

    elif word.isdigit():
        shape = "D0"
    elif len(word)==1 and word in string.punctuation:
        shape = "PUNC"

    return shape

def ner_tag_formatter(input_filename,output_filename, dev_start):

    ner_train = codecs.open(input_filename, 'r+',encoding='latin-1',errors='ignore')
    ner_feature = codecs.open(output_filename, "w+", encoding='latin-1',errors ='ignore')
    line_number = 0
    features = ""
    for line in ner_train:
        line_number += 1

        if dev_start!=0 and line_number>= dev_start:
            ner_feature.write(features)
            ner_feature.close()
            ner_feature =  codecs.open("ner.dev.tmp","w+",encoding='latin-1'
                                                                   '',errors='ignore')
            features = ""
            dev_start = 0

        words = re.split(r'\s+'," ".join(["B_O_S","B_O_S",line.rstrip(),"E_O_S","E_O_S"]))
        #First two words are B_O_S, last one words are E_O_S

        for i, comb_word in enumerate(words[2:-2]):
            curr = i+2
            word_pos_tag = comb_word.rpartition('/')
            tag = word_pos_tag[2]
            word_pos = word_pos_tag[0].rpartition('/')
            word = word_pos[0]
            pos = word_pos[2]

            if words[curr-2] == "B_O_S":
                prev2_word = "B_O_S"
                prev2_tag = ""
                prev2_pos = ""
            else:
                prev2_word_pos_tag = words[curr-2].rpartition('/')
                prev2_tag =  prev2_word_pos_tag[2]
                prev2_word_pos = prev2_word_pos_tag[0].rpartition('/')
                prev2_word = prev2_word_pos[0].lower()
                prev2_pos = prev2_word_pos[2]

            if words[curr-1] == "B_O_S":
                prev1_word = "B_O_S"
                prev1_tag = ""
                prev1_pos = ""
            else:
                prev1_word_pos_tag = words[curr-1].rpartition('/')
                prev1_tag =  prev1_word_pos_tag[2]
                prev1_word_pos = prev1_word_pos_tag[0].rpartition('/')
                prev1_word = prev1_word_pos[0].lower()
                prev1_pos = prev1_word_pos[2]

            features += " ".join([tag , word.lower(),"w_pos:"+pos,"w1_tag:"+prev1_tag,"w1_pos:"+prev1_pos, "w2_tag:"+prev2_tag, "w2_pos:"+prev2_pos,"w_shape:"+wordshape(word),"\n"])
    ner_feature.write(features)
    ner_feature.close()

    return



def main():

    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("trainingfile",nargs = 1, help = "trainingfile input filename")
    parser.add_argument("modelfile",nargs = 1, help = "modelfile output filename")
    parser.add_argument("-h","--DEVFILE", nargs = 1, help = "optional devfile for error rate calculation")
    args = parser.parse_args()


    #open the training_file and convert to features
    #save it to a file and then pass it to the perceptron

    ner_train = codecs.open(args.trainingfile[0], 'r+',encoding='latin-1',errors='ignore')
    train_lines = 0
    i = 0
    for line in ner_train:
        i += 1
    ner_train.close()
    if(args.DEVFILE):
        train_lines = 0
        ner_tag_formatter(args.DEVFILE[0],"ner.dev.fmt", train_lines)
    else:
        train_lines = int(i * .8)


    ner_tag_formatter(args.trainingfile[0],"ner.train.fmt",train_lines)

    #train the perceptron using this formatted file
    if(args.DEVFILE):
        devfile = "ner.dev.fmt"
    else:
        devfile = "ner.dev.tmp"

    perceptron = perceplearn.perceptron_train()
    perceptron.learn("ner.train.fmt",args.modelfile[0],devfile)


#boilerplate for main
if __name__ == '__main__':
    main()

