__author__ = 'Majisha'

import sys
import pickle
import re
import percepclassify
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

def main():

    weights_file = open(sys.argv[1] ,'rb')
    feature_weights = pickle.load(weights_file)
    weights_file.close()

    perceptron = percepclassify.perceptron_classify()

    sys.stdin = codecs.getreader('latin-1')(sys.stdin.detach(), errors='ignore')
    sys.stdout = codecs.getwriter('latin-1')(sys.stdout.detach(), errors='ignore')

    for line in sys.stdin:
        tagged_tokens = []
        tags = []
        pos = []
        tags.append("")
        tags.append("")
        pos.append("")
        pos.append("")

        #line is tokens separated by space

        new_line = " ".join(["B_O_S","B_O_S",line.rstrip(),"E_O_S","E_O_S"])

        tokens = re.split(r'\s+',new_line)

        for i, token in enumerate(tokens[2:-2]):
            curr = i+2
            word_pos = token.rpartition('/')
            word = word_pos[0]
            pos =  word_pos[1]

            prev2_word_pos = tokens[curr-2].rpartition('/')
            prev2_word = prev2_word_pos[0].lower()
            prev2_pos = prev2_word_pos[2]
            prev2_tag = tags[curr-2]
            prev1_word_pos = tokens[curr-1].rpartition('/')
            prev1_word = prev1_word_pos[0].lower()
            prev1_pos = prev1_word_pos[1]
            prev1_tag = tags[curr-1]


            features = " ".join([word.lower(),"w_pos:"+pos,"w1_tag:"+prev1_tag,"w1_pos:"+prev1_pos, "w2_tag:"+prev2_tag, "w2_pos:"+prev2_pos,"w_shape:"+wordshape(word)])
            tags.append(perceptron.classify(features, feature_weights))
            tagged_tokens.append(str(token + "/" + tags[curr]))

        tagged_sequence = " ".join(tagged_tokens)
        sys.stdout.write(tagged_sequence + "\n")
        sys.stdout.flush

    return



#boilerplate for main
if __name__ == '__main__':
    main()