import numpy as np
import random

def manhattan(x, t):
    return np.sum(np.abs(x - t))

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use
 
        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                dists[i_test][i_train] = np.sum(abs(self.train_X[i_train] - X[i_test])) 
        return dists
    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dists[i_test] = np.sum(abs(self.train_X - X[i_test]), axis=1) 
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        def dist(x):
            def dist_over_second_coordinate(t):
                return manhattan(x, t)
            dist_over_second_coordinate = np.vectorize(dist_over_second_coordinate, signature='(n)->()')
            return dist_over_second_coordinate(self.train_X)
        dist = np.vectorize(dist, signature='(m)->(n)')
        dists = dist(X)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        random.seed(42)
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        Y = self.train_y
        Y = np.tile(Y, (num_test, 1))
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            a = Y[i][np.argsort(dists[i])]
            a = a[:self.k]
            classes = [0, 0]
            for class_ in a:
                if (class_):
                    classes[1] += 1
                else:
                    classes[0] += 1
            if classes[1] > classes[0]:
                pred[i] = True
            elif classes[0] > classes[1]:
                pred[i] = False
            else:
                pred[i] = random.randrange(2) == 1
        return pred

    def predict_labels_multiclass(self, dists):
        random.seed(42)
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        Y = self.train_y
        Y = np.tile(Y, (num_test, 1))
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            a = Y[i][np.argsort(dists[i])]
            a = a[:self.k]
            classes = np.zeros(10)
            for class_ in a:
                classes[class_] += 1
            pred[i] = np.argmax(classes)
                
        return pred
