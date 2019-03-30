import os
maze = input("Enter grid number : ")
probability = input("Enter probability : ")
encoder = "./encoder.sh data/maze/grid" + maze + ".txt " + probability + " > mdp" + maze + ".txt"
valueiteration = "./valueiteration.sh mdp" + maze + ".txt " + " > policy" + maze + ".txt"
decoder = "./decoder.sh data/maze/grid" + maze + ".txt " + "policy" + maze + ".txt "+ probability + " > path" + maze + ".txt"
visual = "python3 visualize.py data/maze/grid" + maze + ".txt " + " path" + maze + ".txt"

# print ("Running : ", encoder)
os.system(encoder)

os.system(valueiteration)
os.system(decoder)

print("Path : ")
os.system("cat path" + maze + ".txt")

os.system(visual)