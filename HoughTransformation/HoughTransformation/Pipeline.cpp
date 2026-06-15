#include "pipeline.h"
#include "image.h"
#include "edge_detection.h"
#include "hough.h"

#include <oneapi/tbb/flow_graph.h>
#include <chrono>
#include <iostream>

using namespace oneapi::tbb::flow;

const int SOBEL_THRESHOLD = 100;

const int NUM_THETA = 180;

const int NUM_RHO = 300;

const int HOUGH_THRESHOLD = 100;

void runPipeline(const std::string& inputPath, const std::string& outputPath) {
	graph g;

	// Shared state between phases
	struct PipelineData {
		Image original;
		Image gray;
		Image edges;
		std::vector<std::vector<std::atomic<int>>> accumulator;
		double rhoMax;
	};

	using DataPtr = std::shared_ptr<PipelineData>;


	// Phase 1
	// Load + grayscale
	function_node<std::string, DataPtr> phase1(g, 1, [](const std::string& path) -> DataPtr {
		
		auto data = std::make_shared<PipelineData>();
		
		auto start = std::chrono::high_resolution_clock::now();
		
		data->original = loadImage(path);
		data->gray = toGrayscale(data->original);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 1 (grayscale): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
	});

	// Phase 2
	// Sobel edges detection
	function_node<DataPtr, DataPtr> phase2(g, 1, [](DataPtr data) -> DataPtr {
		
		auto start = std::chrono::high_resolution_clock::now();

		data->edges = applySobel(data->gray, SOBEL_THRESHOLD);
		
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 2 (edges): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
	});

	// Phase 3
	// Hough accumulator
	function_node<DataPtr, DataPtr> phase3(g, 1, [](DataPtr data) -> DataPtr {
		
		auto start = std::chrono::high_resolution_clock::now();

		data->accumulator = buildAccumulator(data->edges, NUM_THETA, NUM_RHO, data->rhoMax);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 3 (accumulator): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
		});


	// Phase 4
	function_node<DataPtr, continue_msg> phase4(g, 1, [&outputPath](DataPtr data) -> continue_msg {

		auto start = std::chrono::high_resolution_clock::now();

		std::vector<Line> lines = findLines(data->accumulator, NUM_THETA, NUM_RHO, data->rhoMax, HOUGH_THRESHOLD);

		Image result = drawLines(data->original, lines);
		saveImage(outputPath, result);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 4 (line count/drawing): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return continue_msg();
		});

	make_edge(phase1, phase2);
	make_edge(phase2, phase3);
	make_edge(phase3, phase4);

	auto totalStart = std::chrono::high_resolution_clock::now();

	phase1.try_put(inputPath);
	g.wait_for_all();

	auto totalEnd = std::chrono::high_resolution_clock::now();
	auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();
	std::cout << "\n[TOTAL] Pipeline: " << totalMs << " ms\n";
}
