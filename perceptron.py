import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes,(self.input_dimensions+1))
        #self.print_weights()
        #raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes,(self.input_dimensions+1)))
        #raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        ones = np.ones((1,X.shape[1]))
        X_added = np.concatenate((ones,X), axis = 0)
        #print(X)
        #print("The shape of X:"+str(X.shape))
        return self.hardlimit(np.dot(self.weights,X_added))

        #raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")
    def hardlimit(self,W):
    	'''if W >=0 :
    		return 1
    	else:
    		return 0'''
    	#print("The shape of weight:"+str(W.shape))
    	#print("The shape of Weigth Shape:"+str(self.weights.shape))
    	y_new = np.zeros((W.shape[0],W.shape[1]))
    	for i in range(0,W.shape[0]):
    		for j in range(0,W.shape[1]):
    			if W[i][j]>0:
    				y_new[i][j] = 1
    			else:
    				y_new[i][j] = 0
    	#print("The shape of predicted:"+str(y_new.shape))
    	#print(y_new)
    	return y_new

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        #raise Warning("You must implement print_weights")

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        ones = np.ones((1,X.shape[1]))
        X_bias = np.concatenate((ones,X),axis = 0)
        #print("X_bias")
        #print(X_bias)
        for epoch in range(0,num_epochs):
            for i in range(0,X.shape[1]):
                a = self.predict(X)
                e = Y - a
                self.weights = self.weights + alpha*np.dot(e[:,i].reshape(self.number_of_classes,1),X_bias[:,i].reshape(self.input_dimensions+1,1).T)



            #a = self.predict(X)
        	#e = Y - a
        	#print(a)
        	#print(Y)
        	#rint(e)
        	#self.weights = self.weights + e*X_bias.T
        	#print("The shape if weigths:"+str(self.weights.shape))
        	#print("The shape of e:"+str(e.shape))
        	#print((np.dot(e,X_bias.T)))
        	#if self.calculate_percent_error(X,Y)
        	#print(self.weights)
        	#print(np.dot(e,X_bias.T))
        	#self.weights = self.weights + np.dot(e,X_bias.T)
        	#self.weights += np.dot(e,X_bias.T)
        	#print(np.dot(self.weights,X_bias))
        	#print(epoch)
        #raise Warning("You must implement train")
        #print("The time:"+str(self.count))
        #print(self.weights)
        #print("The e value")
        #print(e)
        #print("The dot product value")
        #print(np.dot(e,X_bias.T))
        #print("The value of Y")
        #print(Y)
        #print("The value of a")
        #print(a)
        #self.count=self.count+1

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        y_pred = self.predict(X)
        #print(self.weights)
        #print(y_pred[:,0])
        #print(Y[:,0])
        #if y_pred[:,0].all() != y_pred[:,0].all():
        #	print("Entered")
        #if np.array_equal(y_pred[:,0],Y[:,0]):
        #	print("Equal")
        #percent_error = [i for i in range(0,Y.shape[1]) if y_pred[:,i].all()==Y[:,i].all()]
        #for i in range(0, Y.shape[1]):
        #	print(y_pred[:,i])
        #	print(Y[:,i])
        #print("WWWWWWWWWW")
        percent_error = [i for i in range(0,Y.shape[1]) if np.array_equal(Y[:,i],y_pred[:,i])]
        #print(y_pred)
        #print("The slength of output:"+str(len(percent_error)))
        #print(percent_error)
        return (Y.shape[1]-len(percent_error))/Y.shape[1]
        #raise Warning("You must implement calculate_percent_error")

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    model.count = 0
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)
    #print(model.predict(X_train))
    #ones = np.ones((1,X_train.shape[1]))
    #X_train = np.concatenate((ones,X_train),axis=0)
    #print(np.dot(model.weights,X_train))