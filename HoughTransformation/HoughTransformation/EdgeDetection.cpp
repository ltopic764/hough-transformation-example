#include "edge_detection.h"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <cmath>

// Phase 2
// Sobel edge detection
// Sobel uses two 3x3 kernels (Gx and Gy) to compute gradient magnitude
// gradient = sqrt(Gx^2 + Gy^2),  if > threshold, then its edge pixel
Image applySobel(const Image& gray, int threshold) {
	Image edges;
	edges.width = gray.width;
	edges.height = gray.height;
	edges.channels = 1;
	edges.data.resize(gray.width * gray.height, 0);

	// TODO: implement parallel_for over rows
	
	return edges;
}
