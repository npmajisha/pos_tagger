Part I - Averaged Perceptron
Averaged Perceptron will predict the class for 
a feature set and update the weights of the corresponding
features if there is a wrong classification.
The average weights for the features are calculated 
over the iterations.
It uses a maximum iteration of 30, and uses the
best average weight vector found across the 30
iterations. 

To run the training program enter the following command:
python3 perceplearn.py TRAININGFILE MODELFILE [-h DEVFILE]

To run the tagging program enter the following command:

If an optional DEVFILE is provided the average weights obtained
after each iteration is used to test against this devfile.
The error rate is used to determine the quality of the average
feature weights.
If the DEVFILE is not provided, then 20% of the trainingfile is
split into a heldout development data to perform the classification
test.


Part II - Part of Speech Tagging using the Averaged Perceptron


To run the training program enter the following command:
python3 postrain.py TRAININGFILE MODELFILE [-h DEVFILE]

To run the tagging program enter the following command:

The POS tagger uses the averaged perceptron to identify the POS
tags.

The training features are represented as



Part III - Named Entity Recognition using the Averaged Perceptron

To run the program enter the following command:
python3 postag.py TRAININGFILE MODELFILE [-h DEVFILE]

To run the tagging program enter the following command:

Part IV - 

1. What is the accuracy of the part-of-speech tagger?
Answer : Using only 80% of the training data and 20% of it as heldout data

2.What are the precision, recall and F-score for each of the named entity types
 for your named entity recognizer, and what is the overall F-score?
 Answer: