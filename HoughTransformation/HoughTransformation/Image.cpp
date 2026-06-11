#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

#include <iostream>

// Uses stb_image to load image from disc
// returns a pointer to a byte array in memory
// Every pixel takes up 'channels' bytes, if channels=1 its grayscale, if =3 its RGB values
Image loadImage(const std::string& path) {
	Image img;

	// stbi_load automatically sets width, height, channels values from file
	unsigned char* pixels = stbi_load(path.c_str(), &img.width, &img.height, &img.channels, 3);

	if (!pixels) {
		throw std::runtime_error("Failed to load image: " + path);
	}

	img.channels = 3;

	// Copy the pixels from stb memory to vector
	img.data.assign(pixels, pixels + img.width * img.height * img.channels);

	// Free up taken memory
	stbi_image_free(pixels);

	return img;
}

// Phase 1
// grayscale conversion, every pixels in a RGB image is converted to a gray version
Image toGrayscale(const Image& src) {

	// Check if image is already in grayscale
	if (src.channels == 1) {
		std::cout << "Image already in grayscale. Skipping conversion";
		return src;
	}

	Image gray;
	gray.width = src.width;
	gray.height = src.height;
	gray.channels = 1;
	// grayscale image has only 1 byte for pixels
	gray.data.resize(src.width * src.height);

	oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, src.height), 
		// lambda for each row in image
		[&](const oneapi::tbb::blocked_range<int>& range) {

			for (int row = range.begin(); row < range.end(); ++row) {
				for (int col = 0; col < src.width; ++col) {

					// index of the first byte of the pixel in RGB
					int srcIdx = (row * src.width + col) * 3;

					unsigned char r = src.data[srcIdx + 0]; // red
					unsigned char g = src.data[srcIdx + 1]; // green
					unsigned char b = src.data[srcIdx + 2]; // blue

					// grayscale conversion
					unsigned char grayVal = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

					// index in grayscale array, only 1 byte by pixel
					int dstIdx = row * src.width + col;
					gray.data[dstIdx] = grayVal;
				}
			}
		}
	);

	return gray;
}

void saveImage(const std::string& path, const Image& img) {
	int result = 0;

	// Check file extension
	if (path.size() >= 4 && path.substr(path.size() - 4) == ".bmp") {
		// BMP format
		result = stbi_write_bmp(path.c_str(), img.width, img.height, img.channels, img.data.data());
	}
	else if (path.size() >= 4 && path.substr(path.size() - 4) == ".png") {
		// PNG format
		// stride is the number of bytes in a single row
		int stride = img.width * img.channels;
		result = stbi_write_png(path.c_str(), img.width, img.height, img.channels, img.data.data(), stride);
	}
	else {
		throw std::runtime_error("Unsupported image format: " + path + " use .bmp or .png");
	}

	if (!result) {
		throw std::runtime_error("Cannot save image: " + path);
	}
}
