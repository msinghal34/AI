import os
# Testing Deterministic

# for i in range(10,110,10):
#     maze = str(i) + ".txt "
#     encoder = "./encoder.sh data/maze/grid" + maze + "> mdp" + maze
#     valueiteration = "./valueiteration.sh mdp" + maze + "> policy" + maze
#     decoder = "./decoder.sh data/maze/grid" + maze + "policy" + maze + "> path" + maze   
#     print(i)
#     os.system(encoder)
#     os.system(valueiteration)
#     os.system(decoder)
#     visual = "python3 visualize.py data/maze/grid" + maze
#     visual1 = "python3 visualize.py data/maze/grid" + maze + "data/maze/solution" + maze
#     visual2 = "python3 visualize.py data/maze/grid" + maze + "path" + maze
#     os.system("diff data/maze/solution" + maze + "path" + maze)
#     os.system(visual)
#     os.system(visual1)
#     os.system(visual2)
#     print("---------------------------------------------------")

# Testing Stochastic
maze = input("Enter grid number : ")
probability = input("Enter probability : ")
encoder = "./encoder.sh data/maze/grid" + maze + ".txt " + probability + " > mdp" + maze + ".txt"
valueiteration = "./valueiteration.sh mdp" + maze + ".txt " + " > policy" + maze + ".txt"
decoder = "./decoder.sh data/maze/grid" + maze + ".txt " + "policy" + maze + ".txt "+ probability + " > path" + maze + ".txt"
visual = "python3 visualize.py data/maze/grid" + maze + ".txt " + " path" + maze + ".txt"

os.system(encoder)
os.system(valueiteration)
os.system(decoder)
os.system(visual)