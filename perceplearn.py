#Averaged Perceptron
#input the training file and output is model file
#optional devfile
#todo -- split training file in case devfile is not provided
import sys
import re
import argparse
import pickle
import percepclassify
from collections import defaultdict
import time


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
    def update_feature_weights(self,c_iter,label,feature, wt):
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
        
        output = open(modelfile, "w+")

        pickle.dump(feature_weight_vector,output)
        
        output.close()
        return
        
    #this is the learn module
    def learn(self, trainingfile , modelfile,devfile):

        #open the training file
        training_file = open(trainingfile, "r+")

        
        classifier = percepclassify.perceptron_classify()

        
        feature_weights = defaultdict(int)
        avg_feature_weights = defaultdict(int)
        best_avg_vector = defaultdict(int)
        
        prev_error = 1.0
        #preprocessing
        lines = []
        i=0
        
        
        for line in training_file:
            
            tokens = []
            
            tokens = re.split(r'\s+',line.rstrip())
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
                                                
        
        training_file.close()
        
        for label in self.labels:
            self.feature_vector[label] = dict(feature_weights)
            self.avg_feature_vector[label] = dict(avg_feature_weights)
        
        #iterate n times
        c = 1
        for i in range(self.maxIter):
            iter_start = time.time()
            print("Iteration :",i )
            #open the training file
            #training_file = open(trainingfile,"r+")
            #iterate through the training data
            
            #for line in training_file:
            
            for line in lines:
                print("C:",c)
                
                #print("Line:",l)
                
                #get all the separate features, split on whitespace
                #words = re.split(r'\s+' , line.rstrip())
                #words = line
                                             
                calculated_weights = {}
                for label in self.labels:  
                    
                    feature_weights = self.feature_vector[label]
                    weight_calc = 0
                      
##                    for word in words[1:]:
##                        weight_calc += feature_weights[word]
                        
                    weight_calc = sum([feature_weights[word] for word in line[1:]])
                    
                    calculated_weights[label] = weight_calc
                
                classified_label = sorted(calculated_weights,key = calculated_weights.get, reverse = True)[0]

                actual_label = line[0]
                if actual_label != classified_label:                                   
                    
                    for word in line[1:]:
                        self.update_feature_weights(c,classified_label,word,-1.0)
                        self.update_feature_weights(c,actual_label,word,1.0)
                    dict_up_end = time.time()
                    #print("Dictionary", dict_up_end-dict_up_start)
                self.print_feature_weights(1)
                c += 1                                
            iter_end = time.time()
            #print("Iteration time: ",iter_end-iter_start)

                
            #training_file.close()
            
            #check against the dev set
            if (devfile):
                for label in self.labels:    
                    for feature in self.feature_vector[label]:
                        self.update_feature_weights(c-1, label,feature,0)
                
                dev_error = classifier.check_dev_error(devfile, self.avg_feature_vector)
                print("Dev error:",dev_error)
                if prev_error > dev_error:
                    prev_error = dev_error
                    best_avg_vector = dict(self.avg_feature_vector)
                    
        
        if (devfile):
            self.write_weights_file(best_avg_vector,modelfile)        
            return           
                 
        for label in self.labels:    
            for feature in self.feature_vector[label]:
                self.update_feature_weights(c, label,feature,0)
     
        self.print_feature_weights(1)        
        self.write_weights_file(self.avg_feature_vector,modelfile)        
        return


      
def main():
    #parse input arguments  
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("trainingfile",nargs = 1, help = "trainingfile input filename")
    parser.add_argument("modelfile",nargs = 1, help = "modelfile output filename")
    parser.add_argument("-h","--DEVFILE",nargs = "?", help = "optional devfile for error rate calculation")
    args = parser.parse_args()
    
    #initialize the perceptron
    perceptron_c = perceptron_train()
    
    #call the perceptron learn
    if args.DEVFILE:
        perceptron_c.learn(args.trainingfile[0],args.modelfile[0],args.DEVFILE)
    else:
        perceptron_c.learn(args.trainingfile[0],args.modelfile[0], 0 )
     

    
    return  
    
    
   



if __name__ == '__main__':
    main()
