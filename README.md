# K means clustering

This project contains the K means clustering algorithm written in C++ on CPU and on GPU using CUDA. It compares the running times between these two implementations.

## Technologies
- Python 3.8
- C++ v11.3
- CUDA v12.3

## How to use
- In Makefile set up parameters for experiments:
 
    - SEED (to generate random points)
    - POINTS_NUM (number of points to generate (exp of 2))
    - CLUSTERS_NUM (number of clusters)
    - MAX_ITER (maximum number of iterations to find clusters)

- Makefile commands:

    - Run this to generate specified number of points, build and run K means clustering on GPU and CPU. In the terminal one can check running times of both versions. 

```
$ make
```

    - Run this command AFTER make, to plot cisualizations of clustered data. (If the number of points is big (> 4000) it may take long time)

```
$ make plot_data
```

    - Run this to remove object files and executable files 

## Results
    - The running time of the GPU implementation for small amount of points (~100) was similar and even slower than the CPU implementation
    - For ~100 000 points the GPU implementation was about 30 times quicker than the CPU implementation

    - raports available in [raports](/raports/)

Running time of the GPU/CPU implementations:
![Running time of the GPU/CPU implementations](/raports/statistics_10.png)