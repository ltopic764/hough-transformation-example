#include "hough.h"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <atomic>
#include <cmath>
#include <iostream>

const double PI = 3.14159265358979323846;

// Phase 3
// Hough accumulator
std::vector<std::vector<std::atomic<int>>> buildAccumulator(const Image& edges, int numTheta, int numRho, double& rhoMax) {

	if (edges.channels != 1) {
		throw std::runtime_error("buildAccumulator expects an image where cannels = 1");
	}

	// Calculate rhoMax = image diagonal
	rhoMax = std::sqrt(
		static_cast<double>(edges.width) * edges.width +
		static_cast<double>(edges.height) * edges.height
	);

	// Create accumulator
	std::vector<std::vector<std::atomic<int>>> accumulator(numRho);
	for (int r = 0; r < numRho; ++r) {
		// initialize all values to 0
		accumulator[r] = std::vector<std::atomic<int>>(numTheta);
		for (int t = 0; t < numTheta; ++t) {
			accumulator[r][t].store(0);
		}
	}

	double thetaStep = PI / numTheta;

	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, edges.height), [&](const oneapi::tbb::blocked_range<int>& range) {
		for (int y = range.begin(); y < range.end(); ++y) {
			for (int x = 0; x < edges.width; ++x) {

				int idx = y * edges.width + x;

				// if not edges node, continue
				if (edges.data[idx] != 255) {
					continue;
				}

				for (int t = 0; t < numTheta; ++t) {

					double theta = t * thetaStep;

					// Houghs formula
					double rho = x * std::cos(theta) + y * std::sin(theta);

					int rhoIdx = static_cast<int>(
						((rho + rhoMax) / (2.0 * rhoMax)) * numRho
						);

					if (rhoIdx < 0) rhoIdx = 0;
					if (rhoIdx >= numRho) rhoIdx = numRho - 1;

					accumulator[rhoIdx][t].fetch_add(1);
				}
			}
		}
		}
	);

	return accumulator;
}

// Phase 4
// find peaks in accumulator
std::vector<Line> findLines(
	const std::vector<std::vector<std::atomic<int>>>& accumulator,
	int numTheta, int numRho,
	double rhoMax, int threshold) {

	oneapi::tbb::concurrent_vector<Line> found;

	// TODO: parallel_for over accumulator cells
	// find lines based on maxima values

	return std::vector<Line>(found.begin(), found.end());
}

Image drawLines(const Image& original, const std::vector<Line>& lines) {
	Image result = original;
	// TODO: for each line, compute endpoints and draw on result
	return result;
}
