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

void runPipeline(const std::vector<std::pair<std::string, std::string>>& tasks) {
	std::cout << "--- STARTING SEQUENTIAL BATCH ---\n";
	long long tSeq = executeBatch(tasks, false);
	std::cout << "Sequential batch took: " << tSeq << " ms\n\n";

	std::cout << "--- STARTING PARALLEL BATCH ---\n";
	long long tPar = executeBatch(tasks, true);
	std::cout << "Parallel batch took: " << tPar << " ms\n\n";

	double speedup = static_cast<double>(tSeq) / tPar;

	// Generating a summary
	std::ofstream report("batch_report.txt");
	report << "HOUGH TRANSFORMATION BATCH REPORT\n";
	report << "Number of images: " << tasks.size() << "\n\n";
	report << "Total Sequential Time: " << tSeq << " ms\n";
	report << "Total Parallel Time:   " << tPar << " ms\n";
	report << "Total Speedup:         " << speedup << "x\n";
	report.close();

	std::cout << "Report generated: batch_report.txt\n";
}

long long executeBatch(const std::vector<std::pair<std::string, std::string>>& tasks, bool parallelMode) {
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


	// PHASE 1
	function_node<DataPtr, DataPtr> phase1(g, unlimited, [parallelMode](DataPtr data) -> DataPtr {
		data->original = loadImage(data->inputPath);
		data->gray = toGrayscale(data->original, parallelMode);
		return data;
		});

	// PHASE 2
	function_node<DataPtr, DataPtr> phase2(g, unlimited, [parallelMode](DataPtr data) -> DataPtr {
		data->edges = applySobel(data->gray, SOBEL_THRESHOLD, parallelMode);
		return data;
		});

	// PHASE 3
	function_node<DataPtr, DataPtr> phase3(g, unlimited, [parallelMode](DataPtr data) -> DataPtr {
		double diag = std::sqrt(static_cast<double>(data->edges.width) * data->edges.width +
			static_cast<double>(data->edges.height) * data->edges.height);
		data->numRho = static_cast<int>((2.0 * diag) / RHO_BIN_WIDTH);
		data->accumulator = buildAccumulator(data->edges, NUM_THETA, data->numRho, data->rhoMax, data->maxVotes, parallelMode);
		return data;
		});

	// PHASE 4
	function_node<DataPtr, continue_msg> phase4(g, unlimited, [parallelMode](DataPtr data) -> continue_msg {
		int houghThreshold = static_cast<int>(data->maxVotes * HOUGH_THRESHOLD_RATIO);
		std::vector<Line> lines = findLines(data->accumulator, NUM_THETA, data->numRho, data->rhoMax, houghThreshold, parallelMode);

		Image result = drawLines(data->original, lines);
		saveImage(data->outputPath, result);

		std::string edgesPath = data->outputPath.substr(0, data->outputPath.find_last_of('.')) + "_edges.png";
		saveImage(edgesPath, data->edges);

		std::string accPath = data->outputPath.substr(0, data->outputPath.find_last_of('.')) + "_accumulator.png";
		saveAccumulatorImage(accPath, data->accumulator, NUM_THETA, data->numRho, data->maxVotes);

		return continue_msg();
		});

	make_edge(phase1, phase2);
	make_edge(phase2, phase3);
	make_edge(phase3, phase4);

	auto start = std::chrono::high_resolution_clock::now();

	for (const auto& task : tasks) {
		auto data = std::make_shared<PipelineData>();
		data->inputPath = task.first;
		data->outputPath = task.second;
		data->isParallel = parallelMode;
		phase1.try_put(data);

		if (!parallelMode) {
			g.wait_for_all();
		}
	}

	if (parallelMode) {
		g.wait_for_all();
	}

	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}