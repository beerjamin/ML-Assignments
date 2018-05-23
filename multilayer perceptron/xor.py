import sys
import mlp
import matplotlib.pyplot as plt

INPUT_NODES = 2
OUTPUT_NODES = 1
HIDDEN_NODES = 2

#tuneable learning rate and max iterations 
#from command line
LEARNING_RATE = float(sys.argv[1]) 
MAX_ITERATIONS = int(sys.argv[2]) 

class XOR:
    def __init__(self, MLP):
        self.counter = 0
        self.MLP = MLP
        self.epoch = 0

    #training data
    def setXor(self, x):
        if x == 0:
            self.MLP.values[0] = 1
            self.MLP.values[1] = 1
            self.MLP.expectedValues[4] = 0 #expected value of fifth/output node
        elif x == 1:
            self.MLP.values[0] = 0
            self.MLP.values[1] = 1
            self.MLP.expectedValues[4] = 1
        elif x == 2:
            self.MLP.values[0] = 1
            self.MLP.values[1] = 0
            self.MLP.expectedValues[4] = 1
        else:
            self.MLP.values[0] = 0
            self.MLP.values[1] = 0
            self.MLP.expectedValues[4] = 0

    #select training data, update counter
    def setNextTrainingData(self):
        self.setXor(self.counter % 4)
        self.counter += 1

    #@return, epochs: number of times the complete training data has been seen
    #         errors: the mean squared errors 
    def train(self):
        self.errors = []
        self.epochs = []
        for i in range(0, MAX_ITERATIONS): 
          self.setNextTrainingData()
          self.MLP.train()
          #all 4 instances of training data = 1 epoch
          if(self.counter % 4 == 0):
              self.epochs.append(self.epoch)
              self.epoch+=1
              self.errors.append(self.MLP.backprop())
              
        return self.epochs, self.errors

if __name__=="__main__":
    nmlp = mlp.MLP(INPUT_NODES,HIDDEN_NODES,OUTPUT_NODES,LEARNING_RATE)  
    xor = XOR(nmlp)
    
    epochs, errors = xor.train()
    plt.plot(epochs, errors,color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.show()
