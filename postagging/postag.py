__author__ = 'majisha'
import sys
import pickle
import re
import percepclassify
import codecs

def main():

    weights_file = open(sys.argv[1] ,'rb')
    feature_weights = pickle.load(weights_file)
    weights_file.close()

    perceptron = percepclassify.perceptron_classify()

    sys.stdin = codecs.getreader('latin-1')(sys.stdin.detach(), errors='ignore')
    sys.stdout = codecs.getwriter('latin-1')(sys.stdout.detach(), errors='ignore')

    for line in sys.stdin:
        tagged_tokens = []
        #line is tokens separated by space
        line_tokens = re.split(r'\s+',line.rstrip())

        i=0
        no_of_tokens = len(line_tokens)
        if no_of_tokens == 1:
            features = line_tokens[0] + " w_prev:B_O_S" + " w_next:E_O_S"
            classified_label = perceptron.classify(features, feature_weights)
            tagged_tokens.append(str(line_tokens[0] + "/" + classified_label))
        else:
            for token in line_tokens:
                if i==0:
                    features = token + " w_prev:B_O_S" + " w_next:" + line_tokens[i+1]
                elif i == no_of_tokens-1:
                    features = token + " w_prev:" + line_tokens[i-1] + " w_next:E_O_S"
                else:
                    features = token + "w_prev:" + line_tokens[i-1] + " w_next:" + line_tokens[i+1]

                classified_label = perceptron.classify(features, feature_weights)
                tagged_tokens.append(str(token + "/" + classified_label))
        tagged_sequence = " ".join(tagged_tokens)
        sys.stdout.write(tagged_sequence + "\n")
        sys.stdout.flush

    return



#boilerplate for main
if __name__ == '__main__':
    main()