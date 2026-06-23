#pragma once
#include <string>
#include <vector>

// Runs the full Hough pipeline on a single image using TBB flow::graph
// Each phase is a function_node in the graph
void runPipeline(const std::vector<std::pair<std::string, std::string>>& imagePath);
