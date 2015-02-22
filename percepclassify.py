#Perceptron 

import sys
import re
import pickle

class perceptron_classify:
    
    def check_dev_error(self, dev_file , weight_vector):
        dev_text = open(dev_file, 'r+')
        total = 0.0
        count = 0.0
        error = 0.0
        for line in dev_text:
            total += 1
            words = re.split(r'\s+', line.rstrip())
            actual_label = words[0]
            w_line = ' '.join(words[1:])
            
            classified_label = self.classify(w_line, weight_vector)
            if classified_label != actual_label:
                count += 1
                
        
        dev_text.close()
        if total!=0:
            error = count/total
            return error
    
    def classify(self , line , weight_vector):
     
        words = []
        words = re.split(r'\s+', line.rstrip())
        calculated_weights = {}
        for label in weight_vector:
            weights = weight_vector[label]
            weight_calc = 0.0
            for word in words:
                if word in weights:
                    weight_calc += weights[word]
            
            calculated_weights[label] = weight_calc
        classified_label = sorted(calculated_weights,key = calculated_weights.get, reverse = True)[0]
       
        return classified_label

def main():
    
    if len(sys.argv) < 2:
        print("Usage : python3 percepclassify.py weightsfile < input_file")
        return
    
    
    weights_file = open(sys.argv[1] , 'rb')
    feature_weights = pickle.load(weights_file)
    
    weights_file.close()
    
    perceptron = perceptron_classify()
    
     
    for line in sys.stdin:
        classified_label = perceptron.classify(line , feature_weights)
        sys.stdout.write(classified_label + "\n")
        sys.stdout.flush
        

if __name__ == '__main__':
    main()