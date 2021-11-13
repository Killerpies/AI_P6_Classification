"""
naive_bayes.py
-------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import math

import classification_method
import util


class NaiveBayesClassifier(classification_method.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legal_labels):
        super().__init__(legal_labels)
        self.features = []
        self.legal_labels = legal_labels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automatic_tuning = False  # Look at this flag to decide whether to choose k
        # automatically ** use this in your train method **

    def set_smoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, training_data, training_labels, validation_data, validation_labels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in training_data for f in datum.keys()]))

        if self.automatic_tuning:
            k_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            k_grid = [self.k]

        self.train_and_tune(training_data, training_labels,
                            validation_data, validation_labels, k_grid)

    def train_and_tune(self, training_data, training_labels,
                       validation_data, validation_labels, k_grid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in k_grid to choose the smoothing parameter
        that gives the best accuracy on the held-out validation_data.

        training_data and validation_data are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legal_labels.
        """

        best_accuracy_count = -1  # best accuracy so far on validation set

        # Common training - get all counts from training data
        # We only do it once - save computation in tuning smoothing parameter
        common_prior = util.Counter()  # probability over labels
        common_conditional_prob = util.Counter()  # Conditional probability of feature feat being 1
        # indexed by (feat, label)
        common_counts = util.Counter()  # how many time I have seen feature feat with label y
        # whatever inactive or active

        for i in enumerate(training_data):
            i = i[0]
            datum = training_data[i]
            label = training_labels[i]
            common_prior[label] += 1
            for feat, value in list(datum.items()):
                common_counts[(feat, label)] += 1
                if value > 0:  # assume binary value
                    common_conditional_prob[(feat, label)] += 1

        for k in k_grid:  # Smoothing parameter tuning loop!
            prior = util.Counter()
            conditional_prob = util.Counter()
            counts = util.Counter()

            # get counts from common training step
            for key, val in list(common_prior.items()):
                prior[key] += val
            for key, val in list(common_counts.items()):
                counts[key] += val
            for key, val in list(common_conditional_prob.items()):
                conditional_prob[key] += val

            # smoothing:
            for label in self.legal_labels:
                for feat in self.features:
                    conditional_prob[(feat, label)] += k
                    counts[(feat, label)] += 2 * k  # 2 because both value 0 and 1 are smoothed

            # normalizing:
            prior.normalize()
            for x, count in list(conditional_prob.items()):
                conditional_prob[x] = count * 1.0 / counts[x]

            self.prior = prior
            self.conditional_prob = conditional_prob

            # evaluating performance on validation set
            predictions = self.classify(validation_data)
            accuracy_count = [predictions[i] == validation_labels[i] for i in
                              range(len(validation_labels))].count(True)

            print(("Performance on validation set for k=%f: (%.1f%%)" % (
                k, 100.0 * accuracy_count / len(validation_labels))))
            if accuracy_count > best_accuracy_count:
                best_params = (prior, conditional_prob, k)
                best_accuracy_count = accuracy_count
            # end of automatic tuning loop

        self.prior, self.conditional_prob, self.k = best_params

    def classify(self, test_data):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in test_data:
            posterior = self.calculate_log_joint_probabilities(datum)
            guesses.append(posterior.arg_max())
            self.posteriors.append(posterior)
        return guesses

    def calculate_log_joint_probabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        log_joint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legal_labels.
        """
        log_joint = util.Counter()

        for label in self.legal_labels:
            log_joint[label] = math.log(self.prior[label])
            for feat, value in list(datum.items()):
                if value > 0:
                    log_joint[label] += math.log(self.conditional_prob[feat, label])
                else:
                    log_joint[label] += math.log(1 - self.conditional_prob[feat, label])

        return log_joint

    def find_high_odds_features(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        features_odds = []

        for feat in self.features:
            features_odds.append(
                (self.conditional_prob[feat, label1] / self.conditional_prob[feat, label2], feat))
        features_odds.sort()
        features_odds = [feat for val, feat in features_odds[-100:]]

        return features_odds
