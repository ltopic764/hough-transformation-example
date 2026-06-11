#include "pipeline.h"
#include "image.h"
#include "edge_detection.h"
#include "hough.h"

#include <oneapi/tbb/flow_graph.h>
#include <chrono>
#include <iostream>

using namespace oneapi::tbb::flow;

void runPipeline(const std::string& inputPath, const std::string& outputPath) {
	graph g;

	// TODO: define shared state (image data) passed between nodes

	// Phase 1
	// Load + grayscale
	function_node<std::string, Image> phase1(g, 1, [](const std::string& path) -> Image {
		auto start = std::chrono::high_resolution_clock::now();
		// TODO: loadImage + grayscale
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 1 (grayscale): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return {};
	});

	function_node<Image, Image> phase2(g, 1, [](const Image& gray) -> Image {
		auto start = std::chrono::high_resolution_clock::now();
		// TODO: applySobel
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 2 (edges): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return {};
	});

	// Phase 3
	// Phase 4

	make_edge(phase1, phase2);
	// add other edges

	phase1.try_put(inputPath);
	g.wait_for_all();
}
