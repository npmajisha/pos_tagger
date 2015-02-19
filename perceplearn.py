#Averaged Perceptron
#input the training file and output is model file
#todo -- think of dev file
import sys
import re
import argparse
import copy
import pickle
import percepclassify

def update_avg_weights(f ,d ,avg):
    for k , v in d.items():
        avg[k] += f(v)
        
def final_avg_weights(iterations, avg):
    for k, v in avg.items():
        avg[k] /= iterations

class perceptron_train:
    
    def __init__(self):
        self.labels = []
        self.maxIter = 30
    
    def write_weights_file(self, feature_weight_vector, modelfile):
        
        output = open(modelfile,'wb')
        pickle.dump(feature_weight_vector,output,2)
        
        output.close()
##        op = open("weights.txt", 'w+')
##        for label in sorted(self.labels):
##            op.write(label + "\n")
##            for feature in sorted(feature_weight_vector[label]):
##                op.write(feature +" "+ str(feature_weight_vector[label][feature]))
##                op.write("\n")


            
        return
        
    
    def learn(self, trainingfile , devfile):
        
##        output = open("log.txt",'w+')
        classifier = percepclassify.perceptron_classify()
        training_file = open(trainingfile, "r+")
        feature_vector = {}
        avg_feature_vector = {}
        change_counter = {}
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
                change_counter[tokens[0]] = 1
                
            for token in tokens[1:]:
                if token not in feature_weights:
                    feature_weights[token] = 0.0
                    avg_feature_weights[token] = 0.0
        
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
                #print(classified_label)
                
                #print feature weights and average weights
##                output.write("\nIteration :"+str(i+1) +" C iteration : " +str(c)+ '\n')
##                output.write("\n Start of update\n")
##                for label in self.labels:
##                    
##                    output.write("\n"+label + "   Change_counter : " + str(change_counter[label]))
##                    output.write('\n Feature Vector \n')
##                    for key in sorted(feature_vector[label]):
##                        output.write(key + ' ')
##                    output.write("\n")
##                    for key in sorted(feature_vector[label]):
##                        output.write(str(feature_vector[label][key])+ ' ')
##                    output.write('\n Average feature vector \n')
##                    for key in sorted(avg_feature_vector[label]):
##                        output.write(key + ' ')
##                    output.write("\n")
##                    for key in sorted(avg_feature_vector[label]):
##                        output.write(str(avg_feature_vector[label][key])+' ')
                
                if words[0] != classified_label:
                    
                    c_weight_vector = feature_vector[classified_label]
                    c_avg_weight_vector = avg_feature_vector[classified_label]
                    
                    update_avg_weights(lambda x : x * (c - change_counter[classified_label]),c_weight_vector, c_avg_weight_vector)
                    
                    change_counter[classified_label] =  c 
                                        
                    a_weight_vector = feature_vector[words[0]]
                    a_avg_weight_vector = avg_feature_vector[words[0]]                    
                    
                    update_avg_weights(lambda x : x * (c - change_counter[words[0]]), a_weight_vector, a_avg_weight_vector)
                        
                    change_counter[words[0]] = c 
                    
                    for word in words[1:]:
                        c_weight_vector[word] -= 1
                        a_weight_vector[word] += 1
                        
                    #print feature weights and average weights
##                    output.write("\n Vector changed\n")
##                    for label in self.labels:
##                        
##                        output.write("\n"+label+ "   Change_counter : " + str(change_counter[label]))
##                        output.write('\n Feature Vector \n')
##                        for key in sorted(feature_vector[label]):
##                            output.write(key + ' ')
##                        output.write("\n")
##                        for key in sorted(feature_vector[label]):
##                            output.write(str(feature_vector[label][key])+ ' ')
##                        output.write('\n Average feature vector \n')
##                        for key in sorted(avg_feature_vector[label]):
##                            output.write(key + ' ')
##                        output.write("\n")
##                        for key in sorted(avg_feature_vector[label]):
##                            output.write(str(avg_feature_vector[label][key])+' ')
                        
                        
                c += 1                                
                
                
            training_file.close()
            
            #check against the dev set
            if (devfile):
                print(str(classifier.check_dev_error(devfile, c_avg_weight_vector)))
                
        
        for label in self.labels:    
            c_weight_vector = feature_vector[label]
            c_avg_weight_vector = avg_feature_vector[label]
                        
            
            update_avg_weights(lambda x : x * (c - change_counter[label]),c_weight_vector, c_avg_weight_vector)
            final_avg_weights(c - 1,c_avg_weight_vector)
##            output.write("\n Final weights\n") 
##            
##            output.write("\n"+label)
####            output.write('\n Feature Vector \n')
####            for key in sorted(feature_vector[label]):
####                output.write(key + ' ')
####            output.write("\n")
####            for key in sorted(feature_vector[label]):
####                output.write(str(feature_vector[label][key])+ ' ')
##            output.write('\n Average feature vector \n')
##            for key in sorted(avg_feature_vector[label]):
##                output.write(key + ' ')
##            output.write("\n")
##            for key in sorted(avg_feature_vector[label]):
##                output.write(str(avg_feature_vector[label][key])+' ')   
        
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
