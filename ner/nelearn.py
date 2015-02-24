__author__ = 'majisha'

#Named Entity Recognition

import argparse
import perceplearn
import sys
import re
import codecs

def ner_tag_formatter(input_filename,output_filename, dev_start):

    ner_train = codecs.open(input_filename, 'r+',encoding='latin-1',errors='ignore')
    ner_feature = codecs.open(output_filename, "w+", encoding='latin-1',errors ='ignore')
    line_number = 0
    feature = ""
    for line in ner_train:
        line_number += 1

        if dev_start!=0 and line_number>= dev_start:
            ner_feature.write(feature)
            ner_feature.close()
            ner_feature =  codecs.open("ner.dev.tmp","w+",encoding='latin-1'
                                                                   '',errors='ignore')
            feature = ""
            dev_start = 0

        #line is of the form word/postag/nertag
        comb_word_tag = re.split(r'\s+',line.rstrip())


        i=0
        #first token processing
        word_tag = re.split(r'/',comb_word_tag[0])
        prev_pos = ""
        #word_tag = comb_word_tag[0].rpartition('/')

        if len(comb_word_tag) > 1:
            next_tag = re.split(r'/',comb_word_tag[1])
            next_word = "/".join(next_tag[:-2])
            next_pos = next_tag[-2]
            #next_word = comb_word_tag[1].rpartition('/')[0]
        else:
            next_word = "E_O_S"
            next_pos = ""
        #feature label current_word w_prev:prev_word w_next:next_word
        #rpartition returns 3 tuples : word_tag[-1]
        feature += word_tag[-1]+" "+"/".join(word_tag[:-2])+ " c_pos:"+ word_tag[-2]+" prev_ner:"+ " w_prev:B_O_S"+ " prv_pos:"+prev_pos+ " w_next:"+next_word + " nxt_pos:"+ next_pos + "\n"


        for token in comb_word_tag[1:-1]:

            i += 1
            #word_tag = token.rpartition('/')
            word_tag = re.split(r'/',token)
            #prev_tag = comb_word_tag[i-1].rpartition('/')
            prev_tag = re.split(r'/',comb_word_tag[i-1])
            prev_pos = prev_tag[-2]
            next_tag = re.split(r'/',comb_word_tag[i+1])
            next_pos = next_tag[-2]

            feature += word_tag[-1]+" "+ "/".join(word_tag[:-2])+ " c_pos:"+ word_tag[-2]+" prev_ner:"+ prev_tag[-1]+" w_prev:"+"/".join(prev_tag[:-2])+ " prv_pos:"+prev_pos+" w_next:"+"/".join(next_tag[:-2]) + " nxt_pos:"+ next_pos + "\n"

            #pos_feature.write(feature+"\n")

        #last token processing
        if len(comb_word_tag) > 1:
            prev_tag = re.split(r'/',comb_word_tag[-2])
            prev_pos = prev_tag[-2]
            word_tag = re.split(r'/',comb_word_tag[-1])
            feature += word_tag[1]+" "+"/".join(word_tag[:-2])+" c_pos:"+ word_tag[-2]+" prev_ner:"+ prev_tag[-1]+" w_prev:"+"/".join(prev_tag[-2])+" prv_pos:" +prev_pos +" w_next:E_O_S"+ " nxt_pos:"+ "\n"

    ner_feature.write(feature)
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

