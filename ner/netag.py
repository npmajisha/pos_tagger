__author__ = 'majisha'

import sys
import pickle
import re
import percepclassify
import codecs

def main():

    weights_file = open(sys.argv[1], 'rb')
    feature_weights = pickle.load(weights_file)
    weights_file.close()

    perceptron = percepclassify.perceptron_classify()

    sys.stdin = codecs.getreader('latin-1')(sys.stdin.detach(), errors='ignore')
    sys.stdout = codecs.getwriter('latin-1')(sys.stdout.detach(), errors='ignore')


    for line in sys.stdin:

        tagged_tokens = []
        #line is tokens separated by space
        line_tokens = re.split(r'\s+',line.rstrip())
        classified_label = ""
        i=0
        no_of_tokens = len(line_tokens)
        if no_of_tokens == 1:
            word_pos = re.split(r'/',line_tokens[0])
            features = "/".join(word_pos[:-1]) + " c_pos:"+ word_pos[-1] +" prev_ner:"+ classified_label +" w_prev:B_O_S"+ " prv_pos:" + " w_next:E_O_S"+" nxt_pos:"
            classified_label = perceptron.classify(features, feature_weights)
            tagged_tokens.append(str(line_tokens[0] + "/" + classified_label))
        else:
            for token in line_tokens:
                word_pos = re.split(r'/',token)


                if i==0:
                    next_pos = re.split(r'/', line_tokens[i+1])
                    features = "/".join(word_pos[:-1]) + " c_pos:"+ word_pos[-1] +" prev_ner:"+ classified_label +" w_prev:B_O_S"+ " prv_pos:"+ " w_next:" + "/".join(next_pos[:-1])+" nxt_pos:"+next_pos[-1]
                elif i == no_of_tokens-1:
                    prev_pos = re.split(r'/', line_tokens[i-1])
                    features = "/".join(word_pos[:-1]) + " c_pos:"+ word_pos[-1] +" prev_ner:"+ classified_label +" w_prev:"+ "/".join(prev_pos[:-1])+ " prv_pos:"+prev_pos[-1]+ " w_next:E_O_S" +" nxt_pos:"
                else:
                    next_pos = re.split(r'/', line_tokens[i+1])
                    prev_pos = re.split(r'/', line_tokens[i-1])
                    features = "/".join(word_pos[:-1]) + " c_pos:"+ word_pos[-1] +" prev_ner:"+ classified_label +" w_prev:"+ "/".join(prev_pos[:-1])+ " prv_pos:"+prev_pos[-1]+" w_next:" + "/".join(next_pos[:-1])+" nxt_pos:"+next_pos[-1]

                classified_label = perceptron.classify(features, feature_weights)
                tagged_tokens.append(str(token + "/" + classified_label))
        tagged_sequence = " ".join(tagged_tokens)
        sys.stdout.write(tagged_sequence)
        sys.stdout.flush

    return



#boilerplate for main
if __name__ == '__main__':
    main()