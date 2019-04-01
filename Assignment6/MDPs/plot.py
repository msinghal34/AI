import os
import numpy as np
import matplotlib.pyplot as plt

maze = "10"
probabilities = list(np.arange(0,1.05,0.1))
values = [0]*len(probabilities)
counts = []
for prob in probabilities:
	probability = str(prob)
	encoder = "./encoder.sh data/maze/grid" + maze + ".txt " + probability + " > mdp" + maze + ".txt"
	valueiteration = "./valueiteration.sh mdp" + maze + ".txt " + " > policy" + maze + ".txt"
	decoder = "./decoder.sh data/maze/grid" + maze + ".txt " + "policy" + maze + ".txt "+ probability + " > path" + maze + ".txt"
	os.system(encoder)
	os.system(valueiteration)
	os.system(decoder)
	count = 0
	with open("path10.txt", 'r') as f:
		for w in f.readline().split():
			count += 1
	counts.append(count)
plt.plot(probabilities, counts)
plt.xlabel("Probability")
plt.ylabel("Expected number of steps")
plt.title("Probability vs Expected number of steps")
plt.show()