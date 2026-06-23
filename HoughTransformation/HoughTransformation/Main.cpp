#include "pipeline.h"
#include <iostream>
#include <vector>


int main() {
    std::vector<std::pair<std::string, std::string>> tasks = {
        {"input/test.png", "output/res.png"},
        {"input/test1.png", "output/res1.png"},
        {"input/test3.png", "output/res3.png"},
        {"input/test4.png", "output/res4.png"}
    };

    std::cout << "Began Hough transformation...\n";

    runPipeline(tasks);

    std::cout << "Finished.\n";
    return 0;
}
