#pragma once
#include <string>

// Runs the full Hough pipeline on a single image using TBB flow::graph
// Each phase is a function_node in the graph
void runPipeline(const std::string& inputPath, const std::string& outputPath);
