import util
import numpy as np
import sys
import random
import math

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        "*** YOUR CODE HERE ***"
        size = int(len(trainingData))

        weights = [1.0/size for _ in range(size)]

        for k in range(self.boosting_iterations):
            self.classifiers[k].train(trainingData, trainingLabels, weights)
            error = 0.0
            guess = self.classifiers[k].classify(trainingData)
            for j in range(size):
                if guess[j] != trainingLabels[j]:
                    error += 1.0/size
            for j in range(size):
                if guess[j] == trainingLabels[j]:
                    weights[j] *= error/(1.0-error)
            weightsSum = sum(weights)
            weights = [item/weightsSum for item in weights]
            self.alphas[k] = math.log((1.0-error)/error)
        # util.raiseNotDefined()

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        "*** YOUR CODE HERE ***"
        guesses = [0 for i in range(len(data))]
        for i in range(self.boosting_iterations):
            guess = self.classifiers[i].classify(data)
            alpha = self.alphas[i]
            for k in range(len(data)):
                guesses[k] += guess[k]*alpha
        finalGuesses = [util.sign(x) for x in guesses]
        return finalGuesses     
        # util.raiseNotDefined()