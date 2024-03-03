import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

try:
    points = []
    with open("results.txt", 'r') as f:
        points = [l.strip().split() for l in f.readlines()]

    xpoints = [float(p[0]) for p in points]
    ypoints = [float(p[1]) for p in points]
    colors_points = [int(p[2]) for p in points]

    num_of_clusters = int(sys.argv[1])
    colors = cm.rainbow(np.linspace(0, 1, num_of_clusters))

    for i in range(len(xpoints)):
        plt.scatter(xpoints[i], ypoints[i], color=colors[colors_points[i]])

    plt.savefig("points_vis.png")

except FileNotFoundError:
    print("There is no file with results, run K means on CPU first!")

try:
    points = []
    with open("results_cuda.txt", 'r') as f:
        points = [l.strip().split() for l in f.readlines()]

    xpoints = [float(p[0]) for p in points]
    ypoints = [float(p[1]) for p in points]
    colors_points = [int(p[2]) for p in points]

    num_of_clusters = int(sys.argv[1])
    colors = cm.rainbow(np.linspace(0, 1, num_of_clusters))

    for i in range(len(xpoints)):
        plt.scatter(xpoints[i], ypoints[i], color=colors[colors_points[i]])

    plt.savefig("points_vis_cuda.png")

except FileNotFoundError:
    print("There is no file with results, run K means on GPU first!")