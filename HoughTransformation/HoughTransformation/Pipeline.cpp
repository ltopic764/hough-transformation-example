#include "pipeline.h"
#include "image.h"
#include "edge_detection.h"
#include "hough.h"

#include <oneapi/tbb/flow_graph.h>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace oneapi::tbb::flow;

const int SOBEL_THRESHOLD = 100;

const int NUM_THETA = 180;

// width of rho bin in px
const double RHO_BIN_WIDTH = 1.0;

const double HOUGH_THRESHOLD_RATIO = 0.32;

void runPipeline(const std::vector<std::pair<std::string, std::string>>& imagePaths) {
	graph g;

	// Shared state between phases
	struct PipelineData {
		std::string inputPath; // path to original
		std::string outputPath; // path to result

		Image original;
		Image gray;
		Image edges;
		std::vector<std::vector<std::atomic<int>>> accumulator;
		double rhoMax;
		int numRho;
		int maxVotes;
		int otsuThreshold;
		bool isParallel = true;
	};

	using DataPtr = std::shared_ptr<PipelineData>;


	// Phase 1
	// Load + grayscale
	function_node<DataPtr, DataPtr> phase1(g, unlimited, [](DataPtr data) -> DataPtr {
		
		auto start = std::chrono::high_resolution_clock::now();
		
		data->original = loadImage(data->inputPath);
		data->gray = toGrayscale(data->original, data->isParallel);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 1 (grayscale): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
	});

	// Phase 2
	// Sobel edges detection
	function_node<DataPtr, DataPtr> phase2(g, unlimited, [](DataPtr data) -> DataPtr {
		
		auto start = std::chrono::high_resolution_clock::now();

		data->edges = applySobel(data->gray, SOBEL_THRESHOLD, data->isParallel);
		
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 2 (edges): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
	});

	// Phase 3
	// Hough accumulator
	function_node<DataPtr, DataPtr> phase3(g, unlimited, [](DataPtr data) -> DataPtr {
		
		auto start = std::chrono::high_resolution_clock::now();

		// image diagonal
		double diag = std::sqrt(
			static_cast<double>(data->edges.width) * data->edges.width +
			static_cast<double>(data->edges.height) * data->edges.height
		);

		int numRho = static_cast<int>((2.0 * diag) / RHO_BIN_WIDTH);

		data->numRho = numRho;

		data->accumulator = buildAccumulator(data->edges, NUM_THETA, numRho, data->rhoMax, data->maxVotes, data->isParallel);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 3 (accumulator): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
		});


	// Phase 4
	function_node<DataPtr, continue_msg> phase4(g, unlimited, [](DataPtr data) -> continue_msg {

		auto start = std::chrono::high_resolution_clock::now();

		int houghThreshold = static_cast<int>(data->maxVotes * HOUGH_THRESHOLD_RATIO);

		/*std::cout << "  houghThreshold racunat dinamicki: " << houghThreshold
			<< " (= " << (HOUGH_THRESHOLD_RATIO * 100) << "% od maxVotes="
			<< data->maxVotes << ")\n";*/

		std::vector<Line> lines = findLines(data->accumulator, NUM_THETA, data->numRho, data->rhoMax, houghThreshold, data->isParallel);

		// original image with red lines
		Image result = drawLines(data->original, lines);
		saveImage(data->outputPath, result);

		// edge image
		std::string edgesPath = data->outputPath.substr(0, data->outputPath.find_last_of('.')) + "_edges.png";
		saveImage(edgesPath, data->edges);

		// accumulator image
		std::string accPath = data->outputPath.substr(0, data->outputPath.find_last_of('.')) + "_accumulator.png";
		saveAccumulatorImage(accPath, data->accumulator, NUM_THETA, data->numRho, data->maxVotes);


		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 4 (line count/drawing): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";

		return continue_msg();
		});

	make_edge(phase1, phase2);
	make_edge(phase2, phase3);
	make_edge(phase3, phase4);

	auto startTimeBatch = std::chrono::high_resolution_clock::now();

	for (const auto& pair : imagePaths) {
		auto data = std::make_shared<PipelineData>();
		data->inputPath = pair.first;
		data->outputPath = pair.second;

		phase1.try_put(data);
	}

	// wait for graph to finish all images
	g.wait_for_all();

	auto endTimeBatch = std::chrono::high_resolution_clock::now();
	auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeBatch - startTimeBatch).count();

	std::cout << "BATCH PROCESSING FINISHED\n";
	std::cout << "Processed " << imagePaths.size() << " images.\n";
	std::cout << "Total execution time: " << totalTime << " ms\n";

}
