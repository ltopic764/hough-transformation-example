#include "edge_detection.h"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/blocked_range.h>
#include <cmath>
#include <iostream>

// Phase 2
// Sobel edge detection
// Sobel uses two 3x3 kernels (Gx and Gy) to compute gradient magnitude
// gradient = sqrt(Gx^2 + Gy^2),  if > threshold, then its edge pixel
Image applySobel(const Image& gray, int threshold, bool parallel) {

	std::cout << "DEBUG: Sobel ulaz - W:" << gray.width << " H:" << gray.height << " Data size:" << gray.data.size() << std::endl;

	if (gray.data.empty() || gray.width <= 0) {
		throw std::runtime_error("Sobel dobio praznu sliku!");
	}

	// Input image has to be grayscale
	if (gray.channels != 1) {
		throw std::runtime_error("applySobel expects a grayscale image");
	}

	// OTSU
	int otsuThreshold = calculateOtsuThreshold(gray, parallel);
	int finalThreshold = otsuThreshold * 0.9;

	//std::cout << "  Otsu dao prag: " << otsuThreshold << "\n";

	// Output image, same dimensions and also grayscale
	// Values: 0 = not an edge, 255 = edge
	Image edges;
	edges.width = gray.width;
	edges.height = gray.height;
	edges.channels = 1;
	// Initialize all pixels to 0 (not an edge)
	edges.data.resize(gray.width * gray.height, 0);

	auto body = [&](const oneapi::tbb::blocked_range<int>& range) {

		for (int row = range.begin(); row < range.end(); ++row) {
			for (int col = 1; col < gray.width - 1; ++col) {

				// taking tha values of 8 neighbours around the pixel

				int tl = gray.data[(row - 1) * gray.width + (col - 1)]; // top-left
				int tc = gray.data[(row - 1) * gray.width + (col)]; // top-center
				int tr = gray.data[(row - 1) * gray.width + (col + 1)]; // top-right

				int ml = gray.data[(row)*gray.width + (col - 1)]; // mid-left
				// middle pixels not needed, coeficient is 0 in both of the kernels
				int mr = gray.data[(row)*gray.width + (col + 1)]; // mid-right

				int bl = gray.data[(row + 1) * gray.width + (col - 1)]; // bot-left
				int bc = gray.data[(row + 1) * gray.width + (col)]; // bot-cent
				int br = gray.data[(row + 1) * gray.width + (col + 1)]; // bot-right

				// Gx kernel
				int gx = (-1 * tl) + (0 * tc) + (1 * tr)
					+ (-2 * ml) + (0) + (2 * mr)
					+ (-1 * bl) + (0 * bc) + (1 * br);

				// Gy kernel
				int gy = (-1 * tl) + (-2 * tc) + (-1 * tr)
					+ (0 * ml) + (0) + (0 * mr)
					+ (1 * bl) + (2 * bc) + (1 * br);

				// Calculate gradients magnitude
				double magnitude = std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy);

				// Thresholding, see if this is an edge
				int dstIdx = row * edges.width + col;
				edges.data[dstIdx] = (magnitude > finalThreshold) ? 255 : 0;
			}
		}
	};

	if (parallel) {
		oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(1, gray.height-1), body);
	}
	else {
		body(oneapi::tbb::blocked_range<int>(1, gray.height-1));
	}

	// Calculate edges
	int edgeCount = 0;
	for (auto pixel : edges.data) {
		if (pixel == 255) edgeCount++;
	}

	std::cout << "Sobel edge detection finished.\n";
	std::cout << " Edge pixels found: " << edgeCount << "\n";
	
	return edges;
}


int calculateOtsuThreshold(const Image& gray, bool parallel) {
	// Calculate histogram
	std::vector<long long> hist(256, 0);

	if (parallel) {
		// How many of each values from 0-255 there are
		hist = tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, gray.data.size()),
			std::vector<long long>(256, 0),
			[&](const tbb::blocked_range<size_t>& r, std::vector<long long> local_hist) {
				for (size_t i = r.begin(); i < r.end(); ++i) {
					local_hist[gray.data[i]]++;
				}
				return local_hist;
			},
			[](std::vector<long long> a, const std::vector<long long>& b) {
				for (int i = 0; i < 256; ++i) a[i] += b[i];
				return a;
			}
		);
	}
	else {
		// Sequential
		for (unsigned char val : gray.data) {
			hist[val]++;
		}
	}

	long long total = gray.width * gray.height;
	double sum = 0;
	for (int i = 0; i < 256; ++i) sum += i * hist[i];

	double sumB = 0;
	long long wB = 0;
	double maxVar = 0;
	int threshold = 0;

	for (int i = 0; i < 256; ++i) {
		wB += hist[i];
		if (wB == 0) continue;
		long long wF = total - wB;
		if (wF == 0) break;

		sumB += (double)(i * hist[i]);
		double mB = sumB / wB;
		double mF = (sum - sumB) / wF;

		double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);

		if (varBetween > maxVar) {
			maxVar = varBetween;
			threshold = i;
		}
	}
	return threshold;
}
