import sys
import random

seed = int(sys.argv[1])
num_of_points = int(sys.argv[2])
random.seed(seed)
points = [(random.random(), random.random()) for i in range(num_of_points)]

xpoints = [p[0] for p in points]
ypoints = [p[1] for p in points]

with open("points.txt", 'w') as f:
    for p in points:
        f.write(f"{p[0]} {p[1]}\n")