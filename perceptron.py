# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation
import util

PRINT = True


class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legal_labels, max_iterations):
        self.legal_labels = legal_labels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legal_labels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def set_weights(self, weights):
        assert len(weights) == len(self.legal_labels);
        self.weights = weights

    def train(self, training_data, training_labels, validation_data, validation_labels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        self.features = training_data[0].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print(f'Starting iteration {iteration}...')
            for i in range(len(training_data)):
                # print(self.classify([training_data[i]]))
                y = self.classify([training_data[i]])[0]
                # compare yprime to true label y
                if y != training_labels[i]:
                    # true label + training data
                    self.weights[training_labels[i]] += training_data[i]
                    # update weight y. subtract training data
                    self.weights[y] -= training_data[i]

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legal_labels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.arg_max())
        return guesses

    def find_high_weight_features(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        features_weights = []

        # BEGIN SOLUTION METHOD
        features_weights = list(self.weights[label].keys())
        features_weights.sort(key=lambda x: self.weights[label][x])
        features_weights = features_weights[-100:]
        # END SOLUTION

        return features_weights
