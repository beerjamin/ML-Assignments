'''
=========================================
            SUBMITTED BY

            Lorik Mucolli
            Aavash Shrestha
========================================
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import mlab, cm

'''
=======================================================
                      PCA
=======================================================
'''
class PCA:
    def __init__(self, pcs):
        #number of principal components
        self.pcs = pcs 

    def center(self, data):
        self.mean = np.array(np.mean(data, axis=1))
        
        centered = []
        #transpose data as you have to loop through the columns
        for point in np.transpose(data):
            centered.append(np.subtract(point, self.mean))

        #transpose back to convert back to 240x200
        #centered = X_bar
        self.centered = np.transpose(np.array(centered)) 

    def svd(self):
        # C = (1/N) * X_bar * X_bar'
        self.covariance = (1/200) * np.matmul(self.centered, np.transpose(self.centered))
        
        # U, E, U' = SVD(C)
        u, s, u_t = np.linalg.svd(self.covariance)
        
        self.eigenValues = s
        
        # U_m = U[:, 0:pcs]
        self.u = u_t[:, 0:self.pcs]

    def compress(self, data):
        feature = []
        for point in np.transpose(data):
            #f(x) = U_m' * x
            feature.append(np.matmul(np.transpose(self.u), point))
        
        #adding bias in the feature vectors
        for i in range(0, len(feature)):
            feature[i] = np.append(feature[i], 1)
        #self.feature = matrix of feature vectors for each 200 points, pcs x 200
        self.feature = np.transpose(np.array(feature))
        return self.feature

    def reconstruct(self):
        restored = []
        for point in np.transpose(feature):
            # x = mean(240) + U_m(240 x pcs) x feautre_point(pcs x 1) 
            restored.append(np.add(self.mean, np.matmul(self.u, point)))
        self.restored = np.transpose(np.array(restored))
        return self.restored


'''
===========================================================
                FIGURING OUT CUTOFF Ks
===========================================================
'''
def cutoff(eigenvalues, percent):
    sum_1 = sum_2 = k = 0
    percents = []
    for m in range(0,240):
        for k in range(m, 240):
            sum_1 += (eigenvalues[k])**2
        for k in range(0,240):    
            sum_2 += (eigenvalues[k])**2
        percents.append(1-sum_1/sum_2)
    return percents


