'''
Assignment submitted by:
    Lorik Mucolli
    Aavash Shrestha
'''
import math
import numpy
import random

class MLP:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.total_nodes = input_nodes + hidden_nodes + output_nodes
        self.learning_rate = learning_rate

        ''' 
        self.values : activation values of each node
        self.expectedvalues : expected activation value of each node
        '''
        self.values = numpy.zeros(self.total_nodes)
        self.expectedValues = numpy.zeros(self.total_nodes)
        self.thresholds = numpy.zeros(self.total_nodes)
        self.weights = numpy.zeros((self.total_nodes, self.total_nodes))

        '''
        initialize weights 
        and thresholds with random
        values
        '''
        random.seed(10000)
        for i in range(self.input_nodes, self.total_nodes):
            self.thresholds[i] = random.random() / random.random()
            for j in range(i + 1, self.total_nodes):
                self.weights[i][j] = random.random() * 2

    #log sigmoid
    def _sigmoid(self, weight):
        return 1/(1+ math.exp(-weight))

    #derivative of log sigmoid
    def _dsigmoid(self, weight):
        return (weight * (1-weight))  

    def train(self):
        #upper traingular part of weight matrix
        for i in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
            W_i = 0.0
            for j in range(self.input_nodes):
                W_i += self.weights[j][i] * self.values[j]
            W_i -= self.thresholds[i]
            self.values[i] = self._sigmoid(W_i) 

        #lower triangular part of weight matrix 
        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            W_i = 0.0
            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                W_i += self.weights[j][i] * self.values[j]
            W_i -= self.thresholds[i]
            self.values[i] = self._sigmoid(W_i) 

    def backprop(self):
        sumOfSquaredErrors = 0.0

        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            error = self.expectedValues[i] - self.values[i]
            sumOfSquaredErrors += math.pow(error, 2)
            outputErrorGradient = self._dsigmoid(self.values[i]) * error

            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                delta = self.learning_rate * self.values[j] * outputErrorGradient
                self.weights[j][i] += delta
                hiddenErrorGradient = self._dsigmoid(self.values[j]) * outputErrorGradient * self.weights[j][i]

                for k in range(self.input_nodes):
                    delta = self.learning_rate * self.values[k] * hiddenErrorGradient
                    self.weights[k][j] += delta

                delta = self.learning_rate * -1 * hiddenErrorGradient
                self.thresholds[j] += delta
            delta = self.learning_rate * -1 * outputErrorGradient
            self.thresholds[i] += delta
        return sumOfSquaredErrors

         
