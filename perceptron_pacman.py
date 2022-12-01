"""
perceptron_pacman.py
--------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).


Perceptron implementation for apprenticeship learning
"""
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legal_labels, max_iterations):
        PerceptronClassifier.__init__(self, legal_labels, max_iterations)
        self.weights = util.Counter()

    def classify(self, data):
        """
        Data contains a list of (datum, legal moves)

        Datum is a Counter representing the features of each GameState.
        legal_moves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legal_moves in data:
            vectors = util.Counter()
            for l in legal_moves:
                vectors[l] = self.weights * datum[l]  # changed from datum to datum[l]
            guesses.append(vectors.arg_max())
        return guesses

    def train(self, training_data, training_labels, validation_data, validation_labels):
        self.features = list(training_data[0][0]['Stop'].keys())  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(training_data)):
                y = self.classify([training_data[i]])[0]
                # basically same as first problem
                if y != training_labels[i]:
                    self.weights += training_data[i][0][training_labels[i]]
                    self.weights -= training_data[i][0][y]
