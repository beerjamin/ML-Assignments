import sys
import mlp
import matplotlib.pyplot as plt
import numpy as np

INPUT_NODES = 6
OUTPUT_NODES = 1
HIDDEN_NODES = 6

#tune learning rate and max iterations from command line
LEARNING_RATE = float(sys.argv[1]) 
MAX_ITERATIONS = int(sys.argv[2])

class Parity:
    def __init__(self, MLP):
        self.counter = 0
        self.MLP = MLP
        self.epoch = 0
        self.data = np.genfromtxt('parity.txt')

    '''training data in parity.txt
       64 instances of training data
       format: input input input input input input output
       '''
    def setParity(self, x):
        self.MLP.values[0] = self.data[x][0]
        self.MLP.values[1] = self.data[x][1]
        self.MLP.values[2] = self.data[x][2]
        self.MLP.values[3] = self.data[x][3]
        self.MLP.values[4] = self.data[x][4]
        self.MLP.values[5] = self.data[x][5]
        #output node is 13th node
        self.MLP.expectedValues[12] = self.data[x][6] 
          
    #select training data, update counter 
    def setNextTrainingData(self):
        #64 lines
        self.setParity(self.counter % 64)
        self.counter+=1

    #@return, epochs: number of times the complete training data has been seen
    #         errors: the mean squared errors 
    def train(self):
        self.errors = []
        self.epochs = []
        for i in range(0, MAX_ITERATIONS): 
            self.setNextTrainingData()
            self.MLP.train()
            #all 4 instances of training data = 1 epoch
            if(self.counter % 64 == 0):
                 self.epochs.append(self.epoch)
                 self.epoch+=1
                 self.errors.append(self.MLP.backprop())
              
        return self.epochs, self.errors

if __name__=="__main__":
    nmlp = mlp.MLP(INPUT_NODES,HIDDEN_NODES,OUTPUT_NODES,LEARNING_RATE)  
    parity = Parity(nmlp)
    epochs, errors = parity.train()

    plt.plot(epochs, errors,color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.show()
