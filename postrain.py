#POS tag train

import argparse
import perceplearn
import sys
import re
import codecs

def pos_tag_formatter(input_filename,output_filename, dev_start):

    pos_train = codecs.open(input_filename,"r+",'latin-1',errors = 'ignore')
    pos_feature = codecs.open(output_filename, "w+",'latin-1',errors='ignore')

    line_number = 0
    feature = ""
    for line in pos_train:
        line_number += 1

        if dev_start!=0 and line_number>= dev_start:

            pos_feature.write(feature)
            pos_feature.close()
            pos_feature = codecs.open("pos.dev.tmp","w+",'latin-1',errors = 'ignore')
            feature = ""
            dev_start = 0

        #line is of the form word/tag
        comb_word_tag = re.split(r'\s+',line.rstrip())


        i=0
        #first token processing
        word_tag = re.split(r'/',comb_word_tag[0])
        #word_tag = comb_word_tag[0].rpartition('/')

        if len(comb_word_tag) > 1:
            next_word = "/".join(re.split(r'/',comb_word_tag[1])[:-1])
            #next_word = comb_word_tag[1].rpartition('/')[0]
        else:
            next_word = "E_O_S"
        #feature label current_word w_prev:prev_word w_next:next_word
        #rpartition returns 3 tuples : word_tag[-1]
        feature += word_tag[-1]+" "+"/".join(word_tag[:-1])+ " w_prev:B_O_S"+" w_next:"+next_word +"\n"


        #pos_feature.write(feature+"\n")

        for token in comb_word_tag[1:-1]:
            i += 1
            #word_tag = token.rpartition('/')
            word_tag = re.split(r'/',token)
            #prev_tag = comb_word_tag[i-1].rpartition('/')
            prev_tag = re.split(r'/',comb_word_tag[i-1])

            next_tag = re.split(r'/',comb_word_tag[i+1])

            feature += word_tag[-1]+" "+ "/".join(word_tag[:-1])+ " w_prev:"+ "/".join(prev_tag[:-1])+" w_next:"+"/".join(next_tag[:-1])+"\n"

            #pos_feature.write(feature+"\n")

        #last token processing
        if len(comb_word_tag) > 1:
            prev_tag = re.split(r'/',comb_word_tag[-2])
            word_tag = re.split(r'/',comb_word_tag[-1])
            feature += word_tag[-1]+" "+"/".join(word_tag[:-1])+" w_prev:"+"/".join(prev_tag[:-1])+" w_next:E_O_S"+"\n"

    pos_feature.write(feature)
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
