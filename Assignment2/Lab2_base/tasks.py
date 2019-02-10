import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.1 - YOUR CODE HERE
	out_nodes = 2
	alpha = 0.15
	batchSize = 15
	epochs = 20
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(2, 4))
	nn1.addLayer(FullyConnectedLayer(4, 2))
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.2 - YOUR CODE HERE
	out_nodes = 2
	alpha = 0.05
	batchSize = 30
	epochs = 20
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(2, 2))
	nn1.addLayer(FullyConnectedLayer(2, 2))
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

# def taskMnist(x, y=False, z=0):
# 	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
# 	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
# 	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
# 	# Add layers to neural network corresponding to inputs and outputs of given data
# 	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
# 	###############################################
# 	# TASK 2.3 - YOUR CODE HERE

# 	maxAcc = 0.0
# 	alphaMax = 0
# 	batchSizeMax = 0

# 	out_nodes = 10
# 	alpha = [0.01, 0.03, 0.1, 0.2]
# 	batchSize = [5, 15, 25]
# 	epochs = 30

# 	if (y):
# 		print('Network layers : ', '784 X', str(x), 'X', str(z), 'X 10')
# 	else:
# 		print('Network layers : ', '784 X', str(x), 'X 10 ')
# 	for i in range(len(alpha)):
# 		for j in range(len(batchSize)):
# 			nn1 = nn.NeuralNetwork(out_nodes, alpha[i], batchSize[j], epochs)
# 			if (y):
# 				nn1.addLayer(FullyConnectedLayer(784, x)) 
# 				nn1.addLayer(FullyConnectedLayer(x, z))
# 				nn1.addLayer(FullyConnectedLayer(z, 10))
# 				nn1.train(XTrain, YTrain, XVal, YVal, False, False)
# 			else:
# 				nn1.addLayer(FullyConnectedLayer(784, x)) 
# 				nn1.addLayer(FullyConnectedLayer(x, 10))
# 				nn1.train(XTrain, YTrain, XVal, YVal, False, False)
# 			pred, acc  = nn1.validate(XVal, YVal)
# 			print('Alpha: ', str(alpha[i]), 'BatchSize: ', str(batchSize[j]), ' Validation Accuracy ', acc)
# 			if (acc > maxAcc):
# 				maxAcc = acc
# 				alphaMax = alpha[i]
# 				batchSizeMax = batchSize[j]
# 		print('---------------------------------------')
# 	print('---------------------------------------')
# 	print("MAXIMUM: " , maxAcc, alphaMax, batchSizeMax)
# 	# raise NotImplementedError	
# 	###############################################
# 	# nn1.train(XTrain, YTrain, XVal, YVal, False, True)
# 	# pred, acc  = nn1.validate(XTest, YTest)
# 	# print('Test Accuracy ',acc)
# 	# return nn1, XTest, YTest
def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	out_nodes = 10
	alpha = 0.02
	batchSize = 5
	epochs = 15
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(784, 11))
	nn1.addLayer(FullyConnectedLayer(11, 10))
	# raise NotImplementedError	
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	
	XTrain = XTrain[0:500,:,:,:]
	XVal = XVal[0:1000,:,:,:]
	XTest = XTest[0:1000,:,:,:]
	YVal = YVal[0:1000,:]
	YTest = YTest[0:1000,:]
	YTrain = YTrain[0:500,:]
	
	modelName = 'model.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	# # TASK 2.4 - YOUR CODE HERE
	out_nodes = 10
	alpha = 0.02
	batchSize = 50
	epochs = 10

	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(ConvolutionLayer([3, 32, 32], [2, 2], 8, 2))
	nn1.addLayer(AvgPoolingLayer([8, 16, 16], [2, 2], 2))
	nn1.addLayer(FlattenLayer())
	nn1.addLayer(FullyConnectedLayer(512, 10))
	# raise NotImplementedError	
	###################################################
	# return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION


	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=False, saveModel=True, modelName=modelName)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)