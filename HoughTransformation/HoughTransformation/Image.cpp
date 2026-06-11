#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

Image loadImage(const std::string& path) {
	Image img;
	unsigned char* pixels = stbi_load(path.c_str(), &img.width, &img.height, &img.channels, 0);
	if (!pixels) {
		throw std::runtime_error("Failed to load image: " + path);
	}
	img.data.assign(pixels, pixels + img.width * img.height * img.channels);
	stbi_image_free(pixels);
	return img;
}

// Phase 1
// grayscale conversion
Image toGrayscale(const Image& src) {
	Image gray;
	gray.width = src.width;
	gray.height = src.height;
	gray.channels = 1;
	gray.data.resize(src.width * src.height);

	// TODO: implement parallel_for over rows

	return gray;
}

void saveImage(const std::string& path, const Image& img) {
	// TODO
}
