import numpy as np

class linear_regression:
    '''@param features: 1000 rows of feature maps
       @attribute self.targets = 1000 rows of class label vectors
    '''        
    def __init__(self, features):
            self.feature_matrix = features
            target = []
            self.targets = []
            k = 0
            counter = 0
            for i in range(0,1000):
                if(counter==100):
                    counter = 0
                    k+=1
                for j in range(0,10):
                    if (j == k):
                        target.append(1)
                    else:
                        target.append(0)
                counter+=1
                self.targets.append(target)
                target = []
    '''
    @return optimal weight matrix
            phi = feature matrix
            w_opt' = (inv(phi' x phi)) x phi' x targets
    '''
    def optimize(self):
            temp = np.linalg.inv(np.matmul(np.transpose(self.feature_matrix), self.feature_matrix))
            temp = np.matmul(temp, np.transpose(self.feature_matrix))
            w_opt = np.transpose(np.matmul(temp, self.targets))
            return w_opt
                
        
