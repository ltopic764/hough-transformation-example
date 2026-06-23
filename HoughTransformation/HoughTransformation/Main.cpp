#include "pipeline.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::pair<std::string, std::string>> tasks = {
        {"input/test.png", "output/res.png"},
        {"input/test1.png", "output/res1.png"},
        {"input/test3.png", "output/res3.png"},
        {"input/test4.png", "output/res4.png"}
    };

    std::cout << "Began Hough transformation...\n";

    try {
        runPipeline(tasks);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nDone. Press Enter to exit.";
    std::cin.get();
    return 0;

    return 0;
}
