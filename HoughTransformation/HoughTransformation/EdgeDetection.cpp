#include "edge_detection.h"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <cmath>

// Phase 2
// Sobel edge detection
// Sobel uses two 3x3 kernels (Gx and Gy) to compute gradient magnitude
// gradient = sqrt(Gx^2 + Gy^2),  if > threshold, then its edge pixel
Image applySobel(const Image& gray, int threshold) {

	// Input image has to be grayscale
	if (gray.channels != 1) {
		throw std::runtime_error("applySobel expects a grayscale image");
	}

	// Output image, same dimensions and also grayscale
	// Values: 0 = not an edge, 255 = edge
	Image edges;
	edges.width = gray.width;
	edges.height = gray.height;
	edges.channels = 1;
	// Initialize all pixels to 0 (not an edge)
	edges.data.resize(gray.width * gray.height, 0);

	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(1, gray.height - 1), [&](const oneapi::tbb::blocked_range<int>& range) {

		for (int row = range.begin(); row < range.end(); ++row) {
			for (int col = 1; col < gray.width - 1; ++col) {

				// taking tha values of 8 neighbours around the pixel

				int tl = gray.data[(row - 1) * gray.width + (col - 1)]; // top-left
				int tc = gray.data[(row - 1) * gray.width + (col)]; // top-center
				int tr = gray.data[(row - 1) * gray.width + (col + 1)]; // top-right

				int ml = gray.data[(row) *gray.width + (col - 1)]; // mid-left
				// middle pixels not needed, coeficient is 0 in both of the kernels
				int mr = gray.data[(row) *gray.width + (col + 1)]; // mid-right

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
				int magnitude = std::sqrt(gx * gy + gy * gy);

				// Thresholding, see if this is an edge
				int dstIdx = row * edges.width + col;
				edges.data[dstIdx] = (magnitude > threshold) ? 255 : 0;
			}
		}

		});

	// Calculate edges
	int edgeCount = 0;
	for (auto pixel : edges.data) {
		if (pixel == 255) edgeCount++;
	}
	
	return edges;
}
