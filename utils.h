#ifndef UTILS_H
#define UTILS_H

void init_markers(float *points, float *markers, int points_num, int clusters_num);

float dist(float x1, float y1, float x2, float y2);

void points_loader(float *points, int points_num);

void points_recorder(float *points, int *assignments, int points_num);

#endif