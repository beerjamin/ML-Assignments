'''
=========================================
            SUBMITTED BY

            Lorik Mucolli
            Aavash Shrestha
========================================
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import mlab, cm

file ='mfeat-pix.txt'

#load from file and generate numpy array
data = np.genfromtxt(file, delimiter='  ')

#we are using the digit '3'
data = data[600:800, :]

'''
==========================================
				K-MEANS

There are 6 steps of this algorithm
1. Choose a number K of clusters/centroids 
2. Randomly select K points as centroids 
3. Calculate the distance of each point 
    from the centroids
4. Classify other points close to the centroid 
5. Take mean of each cluster, make it new centroid 
6. Repeat steps 3 to 5 until centroids are
    optimized. 
optimized means that the centroids represent
the means of each cluster, the centroid cant
be reassigned anymore
==========================================

'''
class K_Means:
    """
    -- __init__ method with 2 params, k- number of clusters, optimized
    """
    def __init__(self, k=2, optimized=False): 
        self.k = k
        self.optimized=optimized

    def fit(self, data):
    	
    	#the means of the clusters
        #centroids initialized as a dictionary of form centroids = {0: np.array([])....}
        self.centroids = {}
        
        for i in range(self.k): #varying on k, we use that many clusters
            self.centroids[i] = data[i]  #use two initial points as centroids first
        
        while not self.optimized:
           	#clusters
            self.classifications = {}  
            
            for i in range(self.k):
                self.classifications[i] = []
                #initialize first k entries as empty arrays
            
            for featureset in data:
                distances = [np.linalg.norm(np.subtract(featureset,self.centroids[j])) for j in range(0,self.k)]
                # find the distance from each point to the centroid, n
                # featureset datatype represents a single point in the coordinate system
                classification = distances.index(min(distances))
                # populate classification array with indeces of smallest distance. first entry is always 0
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            
           	#caclulate mean for each cluster
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            self.optimized = True

            #see if the algorithm is optimized
            #previous mean == current mean
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
               	
               	if (original_centroid.all() == current_centroid.all()):
               		self.optimized = True


K = 3 # modify k= 1, 2, 3, 200
clf = K_Means(k=K)
clf.fit(data)

'''
========================================================
				VISUALIZATION

We are visualizing how the centroid(codebook vector)
image for each cluster looks like
========================================================
'''
images = []
m=0
#convert vector into proper
#format for visualizing
for i in range(0,K): # K represents the number of selected clusters
	first_dim = []   # initialize first_dim as empty array ready to hold the rows
	k=0                
	for i in range(0,16):
		second_dim = [] # initialize second_dim as empty array ready to hold the columns
		for j in range(0,15):
			second_dim.append(clf.centroids[m][k]) # append 15 entries of codebook vector to second_dim
			k +=1
		first_dim.append(second_dim) # append entries of the column
	m+=1
	images.append(first_dim)

npImages = np.array(images) #convert python list to np.array

norm = cm.colors.Normalize(vmax=1, vmin=0)

fig = plt.figure()

#subplots in the same plot
for i in range(0,K):
	axes = fig.add_subplot(1,3,i+1) #vary parameter accordingly for how many images you want to view
	plt.imshow(npImages[i], cmap='binary', norm=norm)

plt.show()
