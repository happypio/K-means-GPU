#include "utils.h"
#include <iostream>
#include <fstream>
#include <random>

using namespace std;

void init_markers(float *points, float *markers, int points_num, int clusters_num) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distr(0, points_num - 1);
    vector<int> idxs;

    for (int i = 0; i < clusters_num; i++) {
        bool unique;
        int point_idx;
        do {
            unique = true;
            point_idx = distr(gen);

            for (int j = 0; j < idxs.size(); j++)
            {
                if (idxs[j] == point_idx)
                    unique = false;
            }
        }
        while(!unique);

        idxs.push_back(point_idx);

        markers[2 * i] = points[2 * point_idx];
        markers[2 * i + 1] = points[2 * point_idx + 1];
    }

}

float dist(float x1, float y1, float x2, float y2) {
    return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
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
    ofstream outf("results.txt");
    for (int i = 0; i < points_num; i++)
    {
        outf << points[2 * i] << " " << points[2 * i + 1] << " " << assignments[i] << "\n";
    }
    outf.close();
}