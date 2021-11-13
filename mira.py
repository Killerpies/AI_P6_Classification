# mira.py
# -------
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


# Mira implementation
import util

PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legal_labels, max_iterations):
        self.legal_labels = legal_labels
        self.type = "mira"
        self.automatic_tuning = False
        self.C = 0.001
        self.legal_labels = legal_labels
        self.max_iterations = max_iterations
        self.initialize_weights_to_zero()

    def initialize_weights_to_zero(self):
        """Resets the weights of each label to zero vectors"""
        self.weights = {}
        for label in self.legal_labels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def train(self, training_data, training_labels, validation_data, validation_labels):
        """Outside shell to call your method. Do not modify this method."""

        self.features = list(training_data[0].keys())  # this could be useful for your code later...

        if self.automatic_tuning:
            c_grid = [0.002, 0.004, 0.008]
        else:
            c_grid = [self.C]

        return self.train_and_tune(training_data, training_labels, validation_data,
                                   validation_labels, c_grid)

    def train_and_tune(self, training_data, training_labels,
                       validation_data, validation_labels, c_grid):
        """
        This method sets self.weights using MIRA.
        Train the classifier for each value of C in c_grid, then store the weights that give the
        best accuracy on the validation_data.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        # BEGIN SOLUTION METHOD
        print(" Using MIRA update ...")
        best_c, most_hits, best_weights = None, 0, util.Counter()
        for c in c_grid:

            for iteration in range(self.max_iterations):
                for i in enumerate(training_data):
                    i = i[0]
                    score_dictionary = util.Counter()
                    for label in self.legal_labels:
                        score_dictionary[label] = self.weights[label] * training_data[i]

                    predicted_label = score_dictionary.arg_max()
                    if predicted_label != training_labels[i]:

                        # *** YOUR CODE HERE ***
                        util.raise_not_defined()

            # Test for best c.
            accuracy = 0
            list_most_likely_labels = self.classify(validation_data)
            for datum_index in range(len(validation_data)):
                if list_most_likely_labels[datum_index] == validation_labels[datum_index]:
                    accuracy += 1
            if most_hits < accuracy:
                most_hits = accuracy
                best_c = c
                best_weights = self.weights.copy()

        self.C = best_c
        self.weights = best_weights
        # END SOLUTION

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

    def find_high_odds_features(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
                         w_label1 - w_label2

        """
        features_odds = []

        # BEGIN SOLUTION
        for feat in self.features:
            features_odds.append((self.weights[label1][feat] - self.weights[label2][feat], feat))

        features_odds.sort()

        features_odds = [feat for val, feat in features_odds[-100:]]
        return features_odds
        # END SOLUTION
