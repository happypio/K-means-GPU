#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <time.h>


#include <cuda.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

using namespace std;

__constant__ int points_num, clusters_num;

void init_markers(float *points, float *markers, int points_num, int clusters_num) {
    srand(time(NULL));

    int *idxs = (int *)malloc(sizeof(int) * clusters_num);

    for (int i = 0; i < clusters_num; i++) {
        bool unique;
        int point_idx;
        do {
            unique = true;
            point_idx = rand() % points_num;
            for (int j = 0; j < i; j++)
            {
                if (idxs[j] == point_idx)
                    unique = false;
            }
        }
        while(!unique);

        idxs[i] = point_idx;

        markers[2 * i] = points[2 * point_idx];
        markers[2 * i + 1] = points[2 * point_idx + 1];
    }

}

void points_loader(float *points, int points_num)
{
    ifstream inpf("points.txt");
    for (int i = 0; i < points_num; i++)
    {
        inpf >> points[2 * i] >> points[2 * i + 1];
    }
    inpf.close();
}

void points_recorder(float *points, int *assignments, int points_num)
{
    ofstream outf("results_cuda.txt");
    for (int i = 0; i < points_num; i++)
    {
        outf << points[2 * i] << " " << points[2 * i + 1] << " " << assignments[i] << "\n";
    }
    outf.close();
}

__global__ void assign_cluster(float *points, float *markers, int *assignments, float *markers_sums)
{

    extern __shared__ float temp[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    if (tid >= points_num) return;

    // load the markers into shared memory
    if (local_tid < clusters_num) {
        temp[2 * local_tid] = markers[2 * local_tid];
        temp[2 * local_tid + 1] = markers[2 * local_tid + 1];
    }

    

    __syncthreads();

    // load the coordinates
    float x = points[2 * tid];
    float y = points[2 * tid  +1];

    float min_distance = (float)INT32_MAX;
    int assigned_cluster = -1;
    
    for (int c = 0; c < clusters_num; c++)
    {
        float distance = (x - temp[2 * c])*(x - temp[2 * c]) 
                        + (y - temp[2 * c + 1])*(y - temp[2 * c + 1]);

        if (distance < min_distance) {
            assigned_cluster = c;
            min_distance = distance;
        }
    }
    
    // update global assignments
    assignments[tid] = assigned_cluster;

    __syncthreads();

    // for each cluster load the coordinates and perform the reduction
    for (int c = 0; c < clusters_num; c++)
    {
        temp[3 * local_tid] = (assigned_cluster == c) ? x : 0;
        temp[3 * local_tid + 1] = (assigned_cluster == c) ? y : 0;
        temp[3 * local_tid + 2] = (assigned_cluster == c) ? 1 : 0;
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i /= 2)
        {
            if (local_tid < i)
            {  
                int next_local_tid = local_tid + i;
                temp[3 * local_tid] += temp[3 * next_local_tid];
                temp[3 * local_tid + 1] += temp[3 * next_local_tid + 1];
                temp[3 * local_tid + 2] += temp[3 * next_local_tid + 2];
            }
            __syncthreads();
        }
        
        // now update markers sums for this block
        // we keep data as (x,y,count) per cluster and per block
        if (local_tid == 0)
        {
            markers_sums[3 * clusters_num * blockIdx.x + 3 * c] = temp[3 * local_tid];
            markers_sums[3 * clusters_num * blockIdx.x + 3 * c + 1] = temp[3 * local_tid + 1];
            markers_sums[3 * clusters_num * blockIdx.x + 3 * c + 2] = temp[3 * local_tid + 2];
        }

        // wait for threads, in the next step we will need updated markers_sums
        __syncthreads();
    }
}

__global__ void update_markers(float *markers, float *markers_sums, int num_of_partial_sums)
{
    float sum_x = 0;
    float sum_y = 0;
    float count = 0;

    int index = threadIdx.x;
    // iterate over partial sums starting with idx of thread (#threads == #clusters)
    // (increment by number of clusters)
    for(int i = index; i < num_of_partial_sums; i += clusters_num)
    {
        sum_x += markers_sums[3 * i];
        sum_y += markers_sums[3 * i + 1];
        count += markers_sums[3 * i + 2];
    }

    markers[2 * index] = sum_x / count;
    markers[2 * index + 1] = sum_y / count;
}

int main(int argc, const char **argv)
{
    int h_points_num = atoi(argv[1]), h_clusters_num = atoi(argv[2]), h_max_iter = atoi(argv[3]);
    int num_of_threads = 512;
    // ensure that there is enough blocks
    int num_of_blocks = (h_points_num + num_of_threads - 1) / num_of_threads;
    
    int num_of_partial_sums = num_of_blocks * h_clusters_num;

    // allocate memory on host
    float *points  = (float*)malloc(h_points_num * sizeof(float) * 2);
    int *assignments  = (int*)malloc(h_points_num * sizeof(int));
    float *markers = (float*)malloc(h_clusters_num * sizeof(float)  * 2);

    // initialise card

    findCudaDevice(argc, argv);

    checkCudaErrors( cudaMemcpyToSymbol(points_num, &h_points_num, sizeof(h_points_num)));
    checkCudaErrors( cudaMemcpyToSymbol(clusters_num, &h_clusters_num, sizeof(h_clusters_num)));

    // allocate device memory
    float *d_points, *d_markers, *d_markers_sum;
    int *d_assignments;

    checkCudaErrors( cudaMalloc((void**)&d_points, h_points_num * sizeof(float) * 2) );
    checkCudaErrors( cudaMalloc((void**)&d_markers, h_clusters_num * sizeof(float) * 2) );
    checkCudaErrors( cudaMalloc((void**)&d_markers_sum, num_of_blocks * h_clusters_num * sizeof(float) * 3) );

    checkCudaErrors( cudaMalloc((void**)&d_assignments, h_points_num * sizeof(int) ) );

    // load the points coordinates
    points_loader(points, h_points_num);

    // init markers
    init_markers(points, markers, h_points_num, h_clusters_num);

    // run k means and measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // transfer data onto device
    checkCudaErrors( 
        cudaMemcpy(d_points, points, sizeof(float) * h_points_num * 2, cudaMemcpyHostToDevice)
    );
    checkCudaErrors( 
        cudaMemcpy(d_markers, markers, sizeof(float) * h_clusters_num * 2, cudaMemcpyHostToDevice)
    );
    checkCudaErrors( 
        cudaMemcpy(d_assignments, assignments, sizeof(int) * h_points_num, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemset(d_markers_sum, 0, num_of_blocks * h_clusters_num * sizeof(float) * 3)
    );

    cudaEventRecord(start);
    for (int _ = 0; _ < h_max_iter; _++)
    {
        assign_cluster<<<num_of_blocks, num_of_threads, 3 * sizeof(float) * num_of_threads>>>
            (d_points, d_markers, d_assignments, d_markers_sum); 
        cudaDeviceSynchronize();
        
        update_markers<<<1, h_clusters_num>>>(d_markers, d_markers_sum, num_of_partial_sums);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);

    cudaMemcpy(assignments, d_assignments, sizeof(int) * h_points_num, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("Execution of the K means on GPU: %f milliseconds\n", elapsed_time);
    
    // write to the file
    points_recorder(points, assignments, h_points_num);

    free(points);
    free(markers);
    free(assignments);

    cudaFree(d_points);
    cudaFree(d_markers);
    cudaFree(d_markers_sum);
    cudaFree(d_assignments);

    return 0;
}