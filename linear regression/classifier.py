import numpy as np
import matplotlib.pyplot as plt
import l_reg
import pca

class classifier:
    ''' @param data = data read from text file
        @param pcs = principal components(K) for determining number of features
        @attribute train_data = training data sliced from data
        @attribute test_data = test data sliced from data
    '''
    def __init__(self, data, pcs):
        self.pcs = pcs
        counter = 0
        train_data = []
        test_data = []
        for i in range(0, 2000):
            if(counter < 100):
                train_data.append(data[i][:])
            else:
                test_data.append(data[i][:])
            counter+=1
            if(counter == 200):
                counter = 0

        self.train_data = np.transpose(np.array(train_data))
        self.test_data = np.transpose(np.array(test_data))
    
    ''' train model, extract features with PCA
        function approximation with linear regression
        @attribute self.train_features = feature map of train data
        @attribute self.w_opt = optimal weight matrix from linear regresion
        @attribute self.targets = target class label vectors arranged row wise
    '''
    def train(self):
        train_pca = pca.PCA(self.pcs)
        train_pca.center(self.train_data)
        train_pca.svd()
        self.train_features = np.transpose(train_pca.compress(self.train_data))
        LREG = l_reg.linear_regression(self.train_features)
        self.w_opt = LREG.optimize()
        self.targets = LREG.targets
        
    ''' classify train and test data
        use optimal weight matrix to classify data points
        classified vector = w_opt x feature map
        @attribute self.test_features = feature map of test data
        @attribute self.train_classifications = train data classified as class labels
        @attribute self.test_classifications = test data classified as class labels
    '''
    def classify(self):
        test_pca = pca.PCA(self.pcs)
        test_pca.center(self.test_data)
        test_pca.svd()
        self.test_features = np.transpose(test_pca.compress(self.test_data))
        self.train_classifications = []
        self.test_classifications = []

        #looping through the features
        for i in range(0, self.test_features.shape[0]):
            self.test_classifications.append(np.matmul(self.w_opt, self.test_features[i]))
            self.train_classifications.append(np.matmul(self.w_opt, self.train_features[i]))

    ''' @return Average Mean Sqaured Train Error
        train_error = 1/1000 x Summation(Squared_Norm(Classified vector - Target Vector))
    '''
    def mse_train_error(self):
        train_error = 0
        for i in range(0, 1000):
            train_error += (np.linalg.norm(np.subtract(self.train_classifications[i], self.targets[i])))**2
        return (train_error/1000)
    
    ''' @return Average Mean Sqaured Test Error
        train_error = 1/1000 x Summation(Squared_Norm(Classified vector - Target Vector))
    '''       
    def mse_test_error(self):
        test_error = 0
        for i in range(0, 1000):
            test_error += (np.linalg.norm(np.subtract(self.test_classifications[i], self.targets[i])))**2
        return (test_error/1000)
    
    ''' @return Train Missclassification Rate
        miss = 1/1000 x total missclassifications 
        missclassification: index(maximum(classified vector) != class label
    '''
    def miss_train(self):
        cmiss_train = 0
        for i in range(0, 1000):
            max_ind = np.argmax(self.train_classifications[i])
            if(max_ind != np.argmax(self.targets[i])):
                cmiss_train += 1 
        return cmiss_train/1000

    ''' @return Test Missclassification Rate
        miss = 1/1000 x total missclassifications 
        missclassification: index(maximum(classified vector)) != class label
    '''
    def miss_test(self):
        cmiss_test = 0
        for i in range(0, 1000):
            max_ind = np.argmax(self.test_classifications[i])
            if(max_ind != np.argmax(self.targets[i])):
                cmiss_test += 1
        return cmiss_test/1000

if __name__ == "__main__":
    file ='mfeat-pix.txt'

    #load from file and generate numpy array
    data = np.genfromtxt(file, delimiter='  ')

    pcs = []
    mse_train_errors = []
    mse_test_errors= []
    cmiss_train = []
    cmiss_test = []
    for i in range(0, 241):
        cf = classifier(data, i) 
        cf.train()
        cf.classify()
        pcs.append(i)
        mse_train_errors.append(cf.mse_train_error())
        mse_test_errors.append(cf.mse_test_error())
        cmiss_train.append(cf.miss_train())
        cmiss_test.append(cf.miss_test())
    
    #plotting
    plt.plot(pcs, mse_train_errors, color='green', label='Train MSE')
    plt.plot(pcs, mse_test_errors, color='red', label ='Test MSE')
    plt.plot(pcs, cmiss_train, color='green', linestyle=':', label='Train Missclassification')
    plt.plot(pcs, cmiss_test, color='red', linestyle=':', label = 'Test Missclassification')
    plt.legend(fontsize='xx-small')
    plt.show()
    plt.semilogy(pcs, mse_train_errors, color='green', label='Train MSE')
    plt.semilogy(pcs, mse_test_errors, color='red', label ='Test MSE')
    plt.semilogy(pcs, cmiss_train, color='green', linestyle=':', label='Train Missclassification')
    plt.semilogy(pcs, cmiss_test, color='red', linestyle=':', label='Test Missclassification')
    plt.legend(fontsize='xx-small')
    plt.show()
