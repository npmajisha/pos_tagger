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
python3 percepclassify.py MODELFILE < inputfile

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
python3 postag.py MODELFILE < inputfile

The POS tagger uses the averaged perceptron to identify the POS
tags.

The training features are represented as
TAG word w_prev:[previous word] w_next:[next word]

For words at the beginning of a sentence, the previous word is B_O_S
and for words at the end of a sentence the next word in E_O_S


Part III - Named Entity Recognition using the Averaged Perceptron

To run the training program enter the following command:
python3 nelearn.py TRAININGFILE MODELFILE [-h DEVFILE]

To run the tagging program enter the following command:
python3 netag.py MODELFILE < inputfile

The Named Entity recognizer uses the averaged perceptron to identify the NER
tags.

The training features are represented as
TAG word c_pos:[current word pos tag] prev_ner:[previous NER tag] w_prev:[previous word] prv_pos:[previous word pos tag] --
-- w_next:[next word] nxt_pos:[next word pos tag]

Part IV - 

1. What is the accuracy of the part-of-speech tagger?
Answer : Using only 80% of the training data and 20% of it as heldout data, the accuracy of the POS
tagger on the DEVFILE is 93.36%

2.What are the precision, recall and F-score for each of the named entity types
 for your named entity recognizer, and what is the overall F-score?
 Answer:
 
3.What happens if you use your Naive Bayes classifier instead of your perceptron classifier (report performance metrics)?
 Why do you think that is?
 After running the Naive Bayes classifier, the accuracy for POS tag dropped significantly to 22%.
 In POS tagging the tag of a word is greatly dependent on its context, at a fundamental level, it is 
 dependent on its adjacent words.
 Naive Bayes Classification - assumes independence of words in a sentence and classifies on the basis 
 of probabilities (no of occurrences of the word in the document for eg.) , thus the Naive Bayes fails
 to do well in POS tagging where the dependency is extremely important