#Averaged Perceptron
#input the training file and output is model file
#todo -- think of dev file
import sys
import re
import argparse
import copy

class perceptron_train:
    
    def __init__(self):
        self.labels = []
        self.maxIter = 10
    
    def write_weights_file(self, feature_weight_vector, modelfile):
        
        output = open(modelfile,'w+')
        for label in sorted(self.labels):
            output.write(label+" ",)
            
        return
        
    
    def learn(self, trainingfile , devfile):
        
        training_file = open(trainingfile, "r+")
        feature_vector = {}
        avg_feature_vector = {}
        feature_weights = {}
        avg_feature_weights = {}
        #preprocessing
        for line in training_file:
            tokens = []
            
            tokens = re.split(r'\s+',line.rstrip())
            if tokens[0] not in self.labels:
                self.labels.append(tokens[0])
                feature_vector[tokens[0]] = {}
                avg_feature_vector[tokens[0]] = {}
                
            for token in tokens[1:]:
                if token not in feature_weights:
                    feature_weights[token] = 0
                    avg_feature_weights[token] = 0
        
        training_file.close()
        
        for label in self.labels:
            feature_vector[label] = copy.deepcopy(feature_weights)
            avg_feature_vector[label] = copy.deepcopy(avg_feature_weights)
        
        #iterate n times
        c = 1
        for i in range(self.maxIter):
            print("Iteration :",i)
            #open the training file
            training_file = open(trainingfile,"r+")
            #iterate through the training data
            
            for line in training_file:
                words = []
                
                #get all the separate features, split on whitespace
                words = re.split(r'\s+' , line.rstrip())
                                             
                calculated_weights = {}
                
                for label in self.labels:  
                    
                    feature_weights = feature_vector[label]
                    weight_calc = 0
                      
                    for word in words[1:]:
                        weight_calc += feature_weights[word]
                    
                    calculated_weights[label] = weight_calc
                
                classified_label = sorted(calculated_weights,key = calculated_weights.get, reverse = True)[0]
                print(classified_label)
                
                if words[0] != classified_label:
                    c_weight_vector = feature_vector[classified_label]
                    c_avg_weight_vector = avg_feature_vector[classified_label]
                    a_weight_vector = feature_vector[words[0]]
                    a_avg_weight_vector = avg_feature_vector[words[0]]
                    for word in words[1:]:
                        c_weight_vector[word] -= 1
                        c_avg_weight_vector[word] -= c*1
                        a_weight_vector[word] += 1
                        a_avg_weight_vector[word] += c*1
                        
                        
                c += 1                                
                
               
        
        return avg_feature_vector


def main():
    #parse input arguments
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument("trainingfile",nargs = "?", help = "trainingfile input filename")
    parser.add_argument("modelfile",nargs = "?", help = "modelfile output filename")
    parser.add_argument("-h", nargs = "?", help = "optional devfile for error rate calculation")
    args = parser.parse_args()
    
    #initialize the perceptron
    perceptron_c = perceptron_train()
    
    #call the perceptron learn
    if args.h:
        avg_f_w_v = perceptron_c.learn(args.trainingfile,args.h)
    else:
        avg_f_w_v = perceptron_c.learn(args.trainingfile, 0 )
        
    perceptron_c.write_weights_file(avg_f_w_v,args.modelfile)
    
    return

    
    
    
   



if __name__ == '__main__':
    main()
