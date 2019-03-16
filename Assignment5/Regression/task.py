import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	m = X.shape[0]
	n = X.shape[1]
	processedX = np.zeros((m, 0))
	for i in range(n):
		if i == 0:
			processedX = np.append(processedX, np.ones((m, 1)), axis=1)

		elif type(X[0][i]) == str:
			labels = []
			for j in range(m):
				labels.append(X[j,i])
			labels = list(set(labels))
			encoded = one_hot_encode(X[:,i], labels)
			processedX = np.append(processedX, encoded, axis=1)
		
		else:
			mean = np.mean(X[:,i])
			std = np.std(X[:,i])
			ans = (X[:,i] - mean)/std
			processedX = np.append(processedX, np.transpose(np.array([ans])), axis=1)

	return processedX.astype(float), Y.astype(float)

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	grad = np.matmul(np.transpose(X), -2*(Y - np.matmul(X, W))) + 2*_lambda*W
	return grad

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	N = X.shape[0]
	D = X.shape[1]
	XT = np.transpose(X)
	XTX = np.matmul(XT, X)
	XTY = np.matmul(XT, Y)
	weights = np.zeros((D, 1))
	hasnotConverged = True
	while (max_iter != 0) and hasnotConverged:
		max_iter -= 1
		grad = 2*(-XTY + np.matmul(XTX, weights) + _lambda*weights)
		weights = weights -lr*grad
		if (sse(X, Y, weights) <= epsilon):
			hasnotConverged = False
	return weights

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	scores = []
	kFoldsX = np.split(X, k)
	kFoldsY = np.split(Y, k)
	for _lambda in lambdas:
		temp = 0.0
		for i in range(k):
			data = np.concatenate(kFoldsX[:i]+kFoldsX[i+1:])
			labels = np.concatenate(kFoldsY[:i]+kFoldsY[i+1:])
			weights = algo(data, labels, _lambda)
			temp += sse(kFoldsX[i], kFoldsY[i], weights)
		scores.append(temp/k)
	plot_kfold(lambdas, scores)
	return scores

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	
	pass

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [...] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	plot_kfold(lambdas, scores)