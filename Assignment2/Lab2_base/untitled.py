	alpha = [0.1, 0.15, 0.2, 0.25]
	batchSize = [10, 15, 20, 25]
	epochs = [10, 15, 20, 25]
	for i in range(len(alpha)):
		for j in range(len(batchSize)):
			for k in range(len(epochs)):
				nn1 = nn.NeuralNetwork(out_nodes, alpha[i], batchSize[j], epochs[k])
				nn1.addLayer(FullyConnectedLayer(2, 2))
				nn1.addLayer(FullyConnectedLayer(2, 2))
				nn1.train(XTrain, YTrain, XVal, YVal, False, False)
				pred, acc  = nn1.validate(XTrain, YTrain)
				print(str(alpha[i]), str(batchSize[j]), str(epochs[k]), ' Accuracy ', acc)
			print('---------------------------------------')
		print('---------------------------------------')