#pragma once
#include "image.h"
#include <vector>

// A detected line described by (rho, theta)
struct Line {
    double rho;
    double theta;
    int votes;
};

// Phase 3 - build the accumulator
// accumulator[rho_idx][theta_idx] counts votes
// parallelized with parallel_for + atomic increments
std::vector<std::vector<std::atomic<int>>> buildAccumulator(
    const Image& edges, int numTheta, int numRho, double& rhoMax);

// Phase 4 - find lines from accumulator peaks
// uses parallel_for + concurrent_vector to collect results
std::vector<Line> findLines(
    const std::vector<std::vector<std::atomic<int>>>& accumulator,
    int numTheta, int numRho,
    double rhoMax, int threshold);

// Draw detected lines onto a copy of the original image
Image drawLines(const Image& original, const std::vector<Line>& lines);
