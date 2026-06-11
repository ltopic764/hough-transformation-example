#include "hough.h"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <atomic>
#include <cmath>

// Phase 3
// Hough accumulator
std::vector<std::vector<std::atomic<int>>> buildAccumulator(const Image& edges, int numTheta, int numRho, double& rhoMax) {

	// TODO: initialize accumulator with atomis ints set to 0
	// TODO: implement with parallel_for over rows of esge image

	return {};
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
