#pragma once
#include "image.h"

// Phase 2 - edge detection
// Returns a binary edge image (0 = no edge, 255 = edge)
Image applySobel(const Image& gray, int threshold);
int calculateOtsuThreshold(const Image& gray);
