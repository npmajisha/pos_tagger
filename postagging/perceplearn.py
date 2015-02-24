# Averaged Perceptron
#input the training file and output is model file
#optional devfile
#split training file in case devfile is not provided
import sys
import re
import argparse
import pickle
import percepclassify
from collections import defaultdict
import time
import codecs


#perceptron training class
class perceptron_train:
    def __init__(self):
        #List of all the labels that are known
        self.labels = []
        #fixing the iterations to 30
        self.maxIter = 30
        #change_counter for each feature - fast approach to averaging weights across iterations
        self.change_counter = {}
        #For each label there is a feature_vector , which has weights associated
        self.feature_vector = {}
        #Foe each label the avg_feature_vector stores the average weights
        self.avg_feature_vector = {}


    #Below function updates the weights whenever there is a mis-classification
    #Keeping track of the last weight change of a feature/class combination using
    #the change_counter dictionary.
    def update_feature_weights(self, c_iter, label, feature, wt):
        try:
            #diff - the number of iterations the value persisted
            diff = c_iter - self.change_counter[feature][label]
        except KeyError:
            diff = c_iter - 1
            try:
                change_counter = self.change_counter[feature]
            except KeyError:
                self.change_counter[feature] = {}
                change_counter = self.change_counter[feature]

        self.change_counter[feature][label] = c_iter
        #average is calculated using diff
        self.avg_feature_vector[label][feature] += diff * self.feature_vector[label][feature]
        #new weight is assigned to the feature
        self.feature_vector[label][feature] += wt


    #only for debugging purpose
    # def print_feature_weights(self,avg):
    #     print(self.labels[0])
    #     print("\n"+"Feature_weights")
    #     feature_weights =  self.feature_vector[self.labels[0]]
    #     feature_string = ""
    #     for feature in sorted(feature_weights):
    #         feature_string += feature + " "+str(feature_weights[feature])+" "
    #     print(feature_string)
    #     if(avg):
    #         print("Avg_Feature_weights")
    #         feature_weights =  self.avg_feature_vector[self.labels[0]]
    #         feature_string = ""
    #         for feature in sorted(feature_weights):
    #             feature_string += feature + " "+str(feature_weights[feature])+" "
    #         print(feature_string)


    #Writes the learnt model - the average weight vector for the features to the modelfile
    def write_weights_file(self, feature_weight_vector, modelfile):

        output = open(modelfile,"wb")

        pickle.dump(feature_weight_vector, output)

        output.close()
        return

    #this is the learn module
    def learn(self, trainingfile, modelfile, devfile):

        #initializing the weight maps
        feature_weights = defaultdict(int)
        avg_feature_weights = defaultdict(int)
        best_avg_vector = defaultdict(int)

        #setting error rate to 1.0
        prev_error = 1.0

        #Preprocessing the training file to get all the tags
        #and initialize the feature_vectors and avg_feature_vectors

        #open the training file
        training_file = codecs.open(trainingfile, "r+",encoding='latin-1',errors = 'ignore')

        #counter i -to keep track of the lines in the text
        i = 0
        #storing each line of the training file to this List
        lines = []

        for line in training_file:

            tokens = []

            tokens = re.split(r'\s+', line.rstrip())
            lines.append(tokens)
            i += 1

            if tokens[0] not in self.labels:
                self.labels.append(tokens[0])
                self.feature_vector[tokens[0]] = {}
                self.avg_feature_vector[tokens[0]] = {}

            for token in tokens[1:]:
                if token not in feature_weights:
                    feature_weights[token] = 0.0
                    avg_feature_weights[token] = 0.0

        #close the training file
        training_file.close()

        #check if devfile is provided, else split training set
        #currently defaulting to 10% of the total training set

        train_lines = 0
        if (devfile):
            train_lines = i
        else:
            train_lines = int(i * .8)
            #create temporary dev_file
            devfile = "_temp_dev"
            dev_file = codecs.open(devfile, "w+",encoding='latin-1',errors = 'ignore')
            for line in lines[train_lines:]:
                dev_file.write(" ".join(line) + "\n")

            dev_file.close()

        for label in self.labels:
            self.feature_vector[label] = dict(feature_weights)
            self.avg_feature_vector[label] = dict(avg_feature_weights)

        #iterate maxIter times

        c = 1  #counter is to keep track of the averaging

        for i in range(self.maxIter):

            iter_start = time.time()  #timer
            print("Iteration :", i + 1)

            for line in lines[:train_lines - 1]:
                #print("C:",c)

                #Calculated weights for each label stored as hashmap
                calculated_weights = {}

                for label in self.labels:

                    feature_weights = self.feature_vector[label]
                    weight_calc = 0

                    for word in line[1:]:
                        weight_calc += feature_weights[word]

                    calculated_weights[label] = weight_calc

                #class with highest weight for the features is selected
                classified_label = sorted(calculated_weights, key=calculated_weights.get, reverse=True)[0]

                #actuallabel
                actual_label = line[0]

                #update weights in case of wrong-classification
                if actual_label != classified_label:

                    for word in line[1:]:
                        self.update_feature_weights(c, classified_label, word, -1.0)
                        self.update_feature_weights(c, actual_label, word, 1.0)

                #self.print_feature_weights(1)
                c += 1

            #print("Iteration time: ",iter_end-iter_start)

            #Test the performance of the average weight vector using the classifier

            #initialize the perceptron classifier
            classifier = percepclassify.perceptron_classify()

            #After each Iteration test against the Dev set if provided
            if (devfile):

                #update the average weights
                for label in self.labels:
                    for feature in self.feature_vector[label]:
                        self.update_feature_weights(c, label, feature, 0)

                #Call the Perceptron classifier to check for the error rate on dev set
                dev_error = classifier.check_dev_error(devfile, self.avg_feature_vector)
                print("Error:", dev_error)
                #if current error is less than the previous error we set this as the best average vector
                if prev_error > dev_error:
                    prev_error = dev_error
                    best_avg_vector = dict(self.avg_feature_vector)

        #Iteration ends here
            iter_end = time.time()
            print("Time:",iter_end-iter_start)
        if (devfile):
            self.write_weights_file(best_avg_vector, modelfile)
            return

        for label in self.labels:
            for feature in self.feature_vector[label]:
                self.update_feature_weights(c, label, feature, 0)

        #self.print_feature_weights(1)
        #Write the learned model to the modelfile
        self.write_weights_file(self.avg_feature_vector, modelfile)

        return


def main():
    #parse input arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("trainingfile", nargs=1, help="trainingfile input filename")
    parser.add_argument("modelfile", nargs=1, help="modelfile output filename")
    parser.add_argument("-h", "--DEVFILE", nargs=1, help="optional devfile for error rate calculation")

    args = parser.parse_args()

    #initialize the perceptron trainer
    perceptron_c = perceptron_train()

    #call the perceptron learn module
    if args.DEVFILE:
        perceptron_c.learn(args.trainingfile[0], args.modelfile[0], args.DEVFILE[0])
    else:
        perceptron_c.learn(args.trainingfile[0], args.modelfile[0], 0)

    return

#boilerplate for main function

if __name__ == '__main__':
    main()
