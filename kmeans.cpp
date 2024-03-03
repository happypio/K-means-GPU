#include <iostream>
#include <vector>
#include <chrono>
#include "utils.h"

using namespace std;

void k_means(float *points, float *markers, int *assignments, int points_num, int clusters_num, int max_iter)
{
    for (int _ = 0; _ < max_iter; _++)
    {
        // update points assignemnts
        // and keep track of the new mass center

        // avg_x,avg_y and count
        float *new_markers = (float*)malloc(clusters_num * sizeof(float) * 3);
        for (int m=0; m < 3*clusters_num; m++)
            new_markers[m] = 0;

        for (int p = 0; p < points_num; p++)
        {
            float min_distance = (float)INT32_MAX;
            for (int c = 0; c < clusters_num; c++) {
                float distance = dist(points[2 * p], points[2 * p + 1], markers[2 * c], markers[2 * c + 1]);
                if (distance < min_distance){
                    assignments[p] = c;
                    min_distance = distance;
                }
            }

            // update new average
            int c = assignments[p];
            new_markers[3 * c] += points[2 * p];
            new_markers[3 * c + 1] += points[2 * p + 1];
            new_markers[3 * c + 2] += 1;
            
        }

        // update new markers
        for(int c = 0; c < clusters_num; c++)
        {
            int count = new_markers[3 * c  + 2];
            markers[2 * c] = new_markers[3 * c] / count;
            markers[2 * c + 1] = new_markers[3 * c + 1] / count;
        }
        
        free(new_markers);
    }

    
}

int main(int argc, char *argv[])
{
    int points_num = stoi(argv[1]), clusters_num = stoi(argv[2]), max_iter = stoi(argv[3]);
    float *points  = (float*)malloc(points_num * sizeof(float) * 2);
    int *assignments  = (int*)malloc(points_num * sizeof(int));
    float *markers = (float*)malloc(clusters_num * sizeof(float)  * 2);

    // load the points coordinates
    points_loader(points, points_num);

    // init markers
    init_markers(points, markers, points_num, clusters_num);

    // run k means and measure time
    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    k_means(points, markers, assignments, points_num, clusters_num, max_iter);
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float elapsed_time = chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
    cout << "Execution of the K means on CPU: "
         << elapsed_time << " milliseconds\n";
    
    // write to the file
    points_recorder(points, assignments, points_num);

    free(points);
    free(markers);
    free(assignments);

    return 0;
}