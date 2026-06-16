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

void runPipeline(const std::string& inputPath, const std::string& outputPath) {
	graph g;

	// Shared state between phases
	struct PipelineData {
		Image original;
		Image gray;
		Image edges;
		std::vector<std::vector<std::atomic<int>>> accumulator;
		double rhoMax;
		int numRho;
		int maxVotes;
		int otsuThreshold;
		bool isParallel;
	};

	using DataPtr = std::shared_ptr<PipelineData>;


	// Phase 1
	// Load + grayscale
	function_node<DataPtr, DataPtr> phase1(g, 1, [](DataPtr data) -> DataPtr {
		
		auto start = std::chrono::high_resolution_clock::now();
		
		data->gray = toGrayscale(data->original, data->isParallel);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Phase 1 (grayscale): "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< "ms\n";
		return data;
	});

	// Phase 2
	// Sobel edges detection
	function_node<DataPtr, DataPtr> phase2(g, 1, [&outputPath](DataPtr data) -> DataPtr {
		
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
	function_node<DataPtr, DataPtr> phase3(g, 1, [](DataPtr data) -> DataPtr {
		
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
	function_node<DataPtr, continue_msg> phase4(g, 1, [&](DataPtr data) -> continue_msg {

		auto start = std::chrono::high_resolution_clock::now();

		int houghThreshold = static_cast<int>(data->maxVotes * HOUGH_THRESHOLD_RATIO);

		/*std::cout << "  houghThreshold racunat dinamicki: " << houghThreshold
			<< " (= " << (HOUGH_THRESHOLD_RATIO * 100) << "% od maxVotes="
			<< data->maxVotes << ")\n";*/

		std::vector<Line> lines = findLines(data->accumulator, NUM_THETA, data->numRho, data->rhoMax, houghThreshold, data->isParallel);

		// original image with red lines
		Image result = drawLines(data->original, lines);
		saveImage(outputPath, result);

		// edge image
		std::string edgesPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_edges.png";
		saveImage(edgesPath, data->edges);

		// accumulator image
		std::string accPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_accumulator.png";
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

	Image loadedImg = loadImage(inputPath);

	// Sequential execution
	auto dataSeq = std::make_shared<PipelineData>();
	dataSeq->original = loadedImg;
	dataSeq->isParallel = false;

	auto startSeq = std::chrono::high_resolution_clock::now();
	phase1.try_put(dataSeq);
	g.wait_for_all();
	auto endSeq = std::chrono::high_resolution_clock::now();
	long long tSeq = std::chrono::duration_cast<std::chrono::milliseconds>(endSeq - startSeq).count();

	// Parallel execution
	auto dataPar = std::make_shared<PipelineData>();
	dataPar->original = dataSeq->original;
	dataPar->isParallel = true;

	auto startPar = std::chrono::high_resolution_clock::now();
	phase1.try_put(dataPar);
	g.wait_for_all();
	auto endPar = std::chrono::high_resolution_clock::now();
	long long tPar = std::chrono::duration_cast<std::chrono::milliseconds>(endPar - startPar).count();

	// See the difference in execution speed
	double speedup = static_cast<double>(tSeq) / tPar;

	// write to extrenal file
	std::string reportPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_report.txt";
	std::ofstream report(reportPath);
	if (report.is_open()) {
		report << "Result of Hough's transformation\n";
		report << "Input image: " << inputPath << "\n";
		report << "Image dimension: " << loadedImg.width << "x" << loadedImg.height << "\n";
		report << "\n";
		report << "Performance" << "\n";
		report << "Sequential time: " << tSeq << " ms\n";
		report << "Parallel time: " << tPar << " ms\n";
		report << "tSeq/tPar: " << speedup << "x\n";
		report.close();
		std::cout << "Generated report: " << reportPath << "\n";
	}
}
