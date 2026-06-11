#include "pipeline.h"
#include <iostream>

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: hough_tbb <input_image> <output_image>\n";
		std::cout << "Example: hough_tbb images/test1.bmp output/result.bmp\n";
		return 1;
	}

	std::string inputPath = argv[1];
	std::string outputPath = argv[2];

	std::cout << "Running Hough transformation on: " << inputPath << "\n";
	runPipeline(inputPath, outputPath);
	std::cout << "Done" << "\n";

	return 0;
}
