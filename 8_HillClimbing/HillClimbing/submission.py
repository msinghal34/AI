import os
print("Task 1")
os.system("python3 hillclimb.py --file data/tsp10.tsp --r2seed 1 --task 1")
print("Task 2")
os.system("python3 hillclimb.py --file data/berlin52.tsp --r2seed 1 --task 2")
os.system("python3 hillclimb.py --submit --task 2")
print("Task 3")
os.system("python3 hillclimb.py --file data/berlin52.tsp --start_city 1 --task 3")
os.system("python3 hillclimb.py --submit --task 3")
print("Task 4")
os.system("python3 hillclimb.py --file data/berlin52.tsp --start_city 1 --task 4")
os.system("python3 hillclimb.py --submit --task 4")
print("Task 5")
os.system("python3 hillclimb.py --file data/tsp10.tsp --r2seed 1 --task 5")
print("Task 6")
os.system("python3 hillclimb.py --file data/berlin52.tsp --r2seed 1 --task 6")
os.system("python3 hillclimb.py --submit --task 6")
print("Task 7")
os.system("python3 hillclimb.py --file data/berlin52.tsp --start_city 1 --task 7")
os.system("python3 hillclimb.py --submit --task 7")
print("Task 8")
os.system("python3 hillclimb.py --file data/berlin52.tsp --start_city 1 --task 8")
os.system("python3 hillclimb.py --submit --task 8")