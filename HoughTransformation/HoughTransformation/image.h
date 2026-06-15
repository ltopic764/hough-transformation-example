#pragma once
#include <vector>
#include <string>

// Represents a single image (color or grayscale)
struct Image {
    int width;
    int height;
    int channels; // 1 = grayscale, 3 = RGB
    std::vector<unsigned char> data;
};

// Phase 1 functions
Image loadImage(const std::string& path);
Image toGrayscale(const Image& src, bool parallel); // parallelized with parallel_for
void saveImage(const std::string& path, const Image& img);
