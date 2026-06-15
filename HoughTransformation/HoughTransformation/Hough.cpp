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

	// Size of NMS window
	const int neighborhoodSize = 2;

	double thetaStep = PI / numTheta;

	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, numRho), [&](const oneapi::tbb::blocked_range<int>& range) {
		for (int r = range.begin(); r < range.end(); ++r) {
			for (int t = 0; t < numTheta; ++t) {

				// current number of votes for this (rho, theta) cell
				int currentVotes = accumulator[r][t].load();

				// Check if the cell has enough votes
				if (currentVotes <= threshold) {
					continue;
				}

				bool isLocalMax = true;

				for (int dr = -neighborhoodSize; dr <= neighborhoodSize && isLocalMax; ++dr) {
					for (int dt = -neighborhoodSize; dt <= neighborhoodSize; ++dt) {

						// skip self
						if (dr == 0 && dt == 0) continue;

						// 
						int nr = r + dr;
						int nt = t + dt;

						if (nr < 0 || nr >= numRho) continue;
						if (nt < 0 || nt >= numTheta) continue;

						int neighborVotes = accumulator[nr][nt].load();

						// If there exists a neighbor with more votes, current cell is not max
						if (neighborhoodSize > currentVotes) {
							isLocalMax = false;
							break;
						}
					}
				}

				// If cell is max and passes the threshold it is a detected line
				if (isLocalMax) {
					Line line;

					line.theta = t * thetaStep;

					line.rho = (static_cast<double>(r) / numRho) * (2.0 * rhoMax) - rhoMax;

					line.votes = currentVotes;

					found.push_back(line);
				}
			}
		}
		});

	return std::vector<Line>(found.begin(), found.end());
}

Image drawLines(const Image& original, const std::vector<Line>& lines) {
	
	// Copy of the original
	Image result = original;
	
	if (result.channels == 1) {
		std::vector<unsigned char> rgbData(result.width * result.height * 3);
		for (size_t i = 0; i < result.data.size(); ++i) {
			rgbData[i * 3 + 0] = result.data[i]; //R
			rgbData[i * 3 + 1] = result.data[i]; //G
			rgbData[i * 3 + 2] = result.data[i]; //B
		}
		result.data = rgbData;
		result.channels = 3;
	}

	const double L = 1000.0; // line length for drawing (can change)

	for (const Line& line : lines) {

		double cosT = std::cos(line.theta);
		double sinT = std::sin(line.theta);

		// Point on the line closest to (0,0)
		double x0 = line.rho * cosT;
		double y0 = line.rho * sinT;

		int x1 = static_cast<int>(x0 + L * (-sinT));
		int y1 = static_cast<int>(y0 + L * (cosT));

		int x2 = static_cast<int>(x0 - L * (-sinT));
		int y2 = static_cast<int>(y0 - L * (cosT));

		// DDA for line drawing
		int dx = x2 - x1;
		int dy = y2 - y1;
		int steps = std::max(std::abs(dx), std::abs(dy));

		if (steps == 0); continue;

		double xInc = static_cast<double>(dx) / steps;
		double yInc = static_cast<double>(dy) / steps;

		double x = x1;
		double y = y1;

		for (int i = 0; i <= steps; ++i) {

			int px = static_cast<int>(std::round(x));
			int py = static_cast<int>(std::round(y));

			// Check if pixel in image
			if (px >= 0 && px < result.width && py >= 0 && py < result.height) {

				int idx = (py * result.width + px) * 3;

				result.data[idx + 0] = 255;
				result.data[idx + 1] = 0;
				result.data[idx + 2] = 0;
			}

			x += xInc;
			y += yInc;
		}
	}

	return result;
}
