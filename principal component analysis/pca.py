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
        self.mean = np.array(np.mean(data, axis = 1))
        
        centered = []
        #transpose data as you have to loop through the columns
        for point in np.transpose(data):
            centered.append(np.subtract(point, self.mean))

        #transpose back to convert back to 240x200
        #centerd = X_bar
        self.centered = np.transpose(np.array(centered)) 

    def svd(self, data, centered):
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
========================================================
				VISUALIZATION
========================================================
'''

def visualize(data, s_dim1, s_dim2, figs):
        images = []
        m=0
        #convert vector into proper
        #format for visualizing
        for i in range(0,figs): # K represents the number of selected clusters
        	first_dim = []   # initialize first_dim as empty array ready to hold the rows
        	k=0                
        	for i in range(0,16):
        		second_dim = [] # initialize second_dim as empty array ready to hold the columns
        		for j in range(0,15):
        			second_dim.append(data[m][k]) # append 15 entries of codebook vector to second_dim
        			k +=1
        		first_dim.append(second_dim) # append entries of the column
        	m+=1
        	images.append(first_dim)

        npImages = np.array(images) #convert python list to np.array

        fig = plt.figure()

        #subplots
        for i in range(0,figs):
        	axes = fig.add_subplot(s_dim1,s_dim2,i+1)
        	plt.imshow(npImages[i], cmap='binary')
        plt.show()


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

'''
=============================================================
                        MAIN
=============================================================
'''
if __name__ == "__main__":

    file ='mfeat-pix.txt'

    #load from file and generate numpy array
    data = np.genfromtxt(file, delimiter='  ')

    #we are using the digit '7'
    data = np.transpose(data[1400:1600, :])

    # 50% = 2
    # 80% = 6
    # 95% = 28
    # 99% = 152
    # 100% = no k found(infinity)
    pca = PCA(int(sys.argv[1])) #provide pcs as command line arguments
    pca.center(data)
    pca.svd(data, pca.centered)
    feature = pca.compress(data)
    
    #plotting k against covariances
    pcs = cutoff(pca.eigenValues, 0.5)
    x = [y for y in range(0,240)]          
    plt.scatter(x, pcs, marker='.', linewidth='0.2')
    plt.xlabel('Cutoff principal components(k)')
    plt.ylabel('Covariance conserverd (in %)')

    #restore and visualize restored images
    restored = pca.reconstruct()
    visualize(np.transpose(restored),1,5,5)