Part1 - Averaged Perceptron
Averaged Perceptron will predict the class for 
a feature set and update the weights of the corresponding
features if there is a wrong classification.
The average weights for the features are calculated 
over the iterations.
It uses a maximum iteration of 30, and uses the
best average weight vector found across the 30
iterations. 

To run the program enter the following command:
python3 perceplearn.py TRAININGFILE MODELFILE [-h DEVFILE]

If an optional DEVFILE is provided the average weights obtained
after each iteration is used to test against this devfile.
The error rate is used to determine the quality of the average
feature weights.
If the DEVFILE is not provided, then 20% of the trainingfile is
split into a heldout development data to perform the classification
test.


