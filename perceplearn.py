#Averaged Perceptron
#input the training file and output is model file
#todo -- think of dev file
import sys
import re

class perceptron:
    
    def __init__(self):
        self.labels = {}
        
    
    def learn(self,inputfile):
        print("I am learning")
        
        #all labels in the training data    
        label_tags = []
        #feature vector corresponds to each label
        feature_vector = {}
            
        #format of training file--> LABEL feature1 feature2 ... featureN
        
        
        #iterate n times
        for i in range(10):
            print("Iteration :",i)
            #open the training file
            training_file = open(inputfile,"r+")
            #iterate through the training data
            for line in training_file:
                features = []
                
                #get all the separate features, split on whitespace
                features = re.split(r'\s+' , line.rstrip())
                
                #check if no labels have been found yet, if yes then assign the same label
                if len(self.labels) == 0:
                    #to do -assign the label itself
                    print("First example")
                
                
                #training data Begins with label    
                if features[0] not in self.labels:
                    self.labels[features[0]] = 1
                
                calculated_weights = {}
                for label in self.labels:  
                    
                    weight_calc = 0
                      
                    for feature in features[1:]:
                        
                        if feature not in feature_vector:
                            
                            weight_vectors = {}
                            for tag in self.labels:
                                weight_vectors[tag] = 0
                            feature_vector[feature] = weight_vectors
                            
                        else:
                            #feature is existing in the feature - calculate weight of all the classes
                            weight_vectors = feature_vector[feature]
                            if label in weight_vectors:
                                weight_calc += weight_vectors[label]
                            else:
                                weight_vectors[label] = 0
                    
                    calculated_weights[label] = weight_calc
                
                classified_label = sorted(calculated_weights,key = calculated_weights.get, reverse = True)[0]
                print(classified_label)
                if features[0] != classified_label:
                    for feature in features[1:]:
                        weight_vector = feature_vector[feature]
                        weight_vector[classified_label] -= 1
                        weight_vector[features[0]] += 1
            
            
        
        return


def main():
    #input argument is training filename
    
    #initialize the perceptron
    perceptron_c = perceptron()
    
    #call the perceptron learn
    perceptron_c.learn(sys.argv[1])
    
    return

    
    
    
   



if __name__ == '__main__':
    main()
