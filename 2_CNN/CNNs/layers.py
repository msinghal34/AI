import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = (np.dot(X, self.weights) + self.biases)
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# print('Backward FC ',self.weights.shape)
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		delSigma = delta*derivative_sigmoid(self.data) # n X out_nodes
		new_delta = np.dot(delSigma, self.weights.transpose()) # n X in_nodes
		self.weights -= lr*(np.dot((activation_prev.transpose()), delSigma)) # in_nodes X out_nodes
		self.biases -= lr*sum(delSigma) # 1 X out_nodes
		return new_delta
		# raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1, self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		temp = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for z in range(n):
			for k in range(0, self.out_depth):
				for i in range(0, self.in_row - self.filter_row + 1, self.stride):
					for j in range(0, self.in_col - self.filter_col + 1, self.stride):
						temp[z, k, (i // self.stride), (j // self.stride)] \
						= sum(sum(sum(X[z, 0:self.in_depth, i:i+self.filter_row, j:j+self.filter_col] * self.weights[k])))
				temp[z, k, 0:, 0:] = temp[z, k, 0:, 0:] + self.biases[k]
		self.data = temp
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# print('Backward CN ',self.weights.shape)
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# print("-------------------------")
		# print(activation_prev.shape)
		# print(delta.shape)
		# print(self.data.shape)
		# print(self.weights.shape)
		# print(self.biases.shape)
		
		delSigma = delta*derivative_sigmoid(self.data)
		new_delta = np.zeros(activation_prev.shape)

		# New Delta Calculation
		for z in range(delta.shape[0]):
			for k in range(delta.shape[1]):
				for i in range(delta.shape[2]):
					for j in range(delta.shape[3]):
						new_delta[z, :, i*self.stride:i*self.stride+self.filter_row, j*self.stride:j*self.stride+self.filter_col] += \
						delSigma[z,k,i,j]*self.weights[k, :, :, :]

		# Weights Update
		gradWeight = np.zeros(self.weights.shape)
		for z in range(delta.shape[0]):
			for k in range(delta.shape[1]):
				for i in range(delta.shape[2]):
					for j in range(delta.shape[3]):
						gradWeight[k, :, :, :] += delSigma[z, k, i, j]*\
						activation_prev[z, :, i*self.stride:i*self.stride+self.filter_row, j*self.stride:j*self.stride+self.filter_col]
		self.weights -= lr*gradWeight

		# Bias Update
		gradBias = np.sum(np.sum(np.sum(delSigma, axis=0), axis = 1), axis = 1)
		self.biases -= lr*gradBias

		# Return New_Delta
		return new_delta
		# raise NotImplementedError
		###############################################

class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward AvgPoolingLayer ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		temp = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		for z in range(n):
			for k in range(0, self.out_depth):
				for i in range(0, self.in_row - self.filter_row + 1, self.stride):
					for j in range(0, self.in_col - self.filter_col + 1, self.stride):
						temp[z, k, i // self.stride, j // self.stride] = np.average(X[z, k, i:i+self.stride, j:j+self.stride])
		return temp
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# print('Backward AvgPoolingLayer ')
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		avgFac = self.stride*self.stride
		delta /= avgFac
		new_delta = np.repeat(np.repeat(delta, self.stride, axis = 3), self.stride, axis = 2)
		return new_delta
		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # print('Forward FlattenLayer ')
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
    	# print('Backward FlattenLayer ')
    	return delta.reshape(self.in_batch, self.r, self.c, self.k)

# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))