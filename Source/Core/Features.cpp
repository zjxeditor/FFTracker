//
// Image features extraction.
//

#include "Features.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"

namespace CSRT {

using namespace std;

FeaturesExtractor GFeatsExtractor;

// Feature memory arena management.
static MemoryArena *GFeatArenas = nullptr;

bool FeaturesExtractor::SingleInstance = false;

FeaturesExtractor::FeaturesExtractor() : initialized(false) {
	if (SingleInstance) {
		Critical("Only one \"FeaturesExtractor\" can exists.");
		return;
	}
	SingleInstance = true;
}

FeaturesExtractor::~FeaturesExtractor() {
	if (GFeatArenas) {
		delete[] GFeatArenas;
		GFeatArenas = nullptr;
	}
	SingleInstance = false;
}

void FeaturesExtractor::Initialize() {
	if (initialized) return;
	initialized = true;
	GFeatArenas = new MemoryArena[MaxThreadIndex()];
}

//
// HOG Features
//

void FeaturesExtractor::GetFeaturesHOG(const Mat &img, std::vector<MatF> &features, int binSize, MemoryArena *targetArena) {
	ComputeHOG32D(img, features, binSize, 1, 1, targetArena);
}

void FeaturesExtractor::ComputeHOG32D(const Mat& img, std::vector<MatF> &features, int sbin, int padx, int pady, MemoryArena *targetArena) {
	if (padx < 0 || pady < 0 || img.Channels() != 3)
		Critical("FeaturesExtractor::ComputeHOG32D: cannot computer hog feature on current image.");

	// Convert to MatF.
	MatF imgM(&GFeatArenas[ThreadIndex]);
	img.ToMatF(imgM, 1.0f / 255.0f);

	static const int dimHOG = HOG_FEATURE_COUNT;		// Feature dimension.
	static const float eps = 0.0001f;					// Epsilon to avoid division by zero.
	static const int numOrient = 18;					// Number of orientations.
	// Unit vectors to compute gradient orientation.
	static const float uu[9] = { 1.000f, 0.9397f, 0.7660f, 0.5000f, 0.1736f, -0.1736f, -0.5000f, -0.7660f, -0.9397f };
	static const float vv[9] = { 0.000f, 0.3420f, 0.6428f, 0.8660f, 0.9848f,  0.9848f,  0.8660f,  0.6428f,  0.3420f };

	// Mat size.
	const Vector2i imageSize(img.Cols(), img.Rows());
	// Block size.
	int bW = (int)std::floor((float)imageSize.x / sbin);
	int bH = (int)std::floor((float)imageSize.y / sbin);
	const Vector2i blockSize(bW, bH);
	// Size of HOG features.
	int oW = std::max(blockSize.x - 2, 0) + 2 * padx;
	int oH = std::max(blockSize.y - 2, 0) + 2 * pady;
	const Vector2i outSize(oW, oH);
	// Size of visible.
	const Vector2i visibleSize = blockSize * sbin;

	// Initialize historgram, norm, output feature matrices.
	MatF histM(blockSize.y, blockSize.x*numOrient, &GFeatArenas[ThreadIndex]);
	MatF normM(blockSize.y, blockSize.x, &GFeatArenas[ThreadIndex]);
	MatF diffM(visibleSize.y, visibleSize.x * 2, &GFeatArenas[ThreadIndex]);
	features.clear();
	features.resize(dimHOG, MatF(targetArena));
	for (int i = 0; i < dimHOG; ++i) features[i].Reshape(outSize.y, outSize.x);

	// Get the stride of each matrix.
	const size_t imStride = imgM.Cols();
	const size_t diffStride = diffM.Cols();
	const size_t histStride = histM.Cols();
	const size_t normStride = normM.Cols();
	const size_t featStride = outSize.x;

	// Calculate the zero offset.
	const float* im = imgM.Data();
	float* const diff = diffM.Data();
	float* const hist = histM.Data();
	float* const norm = normM.Data();

	// Calculate each pixels gradent and best matched direction.
	ParallelFor([&](int64_t y) {
		++y;
		// RGB data is in interleaved format
		const float *src = im + y * imStride + 3;
		float *dest = diff + y * diffStride + 2;
		for (int x = 1; x < visibleSize.x - 1; ++x) {
			// Red image channel
			float dy = *(src + imStride) - *(src - imStride);
			float dx = *(src + 3) - *(src - 3);
			float v = dx * dx + dy * dy;
			++src;
			// Green image channel
			float dyg = *(src + imStride) - *(src - imStride);
			float dxg = *(src + 3) - *(src - 3);
			float vg = dxg * dxg + dyg * dyg;
			++src;
			// Blue image channel
			float dyb = *(src + imStride) - *(src - imStride);
			float dxb = *(src + 3) - *(src - 3);
			float vb = dxb * dxb + dyb * dyb;
			++src;

			// Pick the channel with the strongest gradient
			if (vg > v) { v = vg; dx = dxg; dy = dyg; }
			if (vb > v) { v = vb; dx = dxb; dy = dyb; }
			v = sqrt(v);

			// Snap to one of the 18 orientations
			float best_dot = 0.0f;
			int best_o = 0;
			for (int o = 0; o < numOrient / 2; ++o) {
				float dot = uu[o] * dx + vv[o] * dy;
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				} else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o + numOrient / 2;
				}
			}

			*(dest++) = v;
			*(dest++) = (float)best_o;
		}
	}, visibleSize.y - 2, 32);

	// Calculate HOG histogram.
	for (int y = 1; y < visibleSize.y - 1; ++y) {
		for (int x = 1; x < visibleSize.x - 1; ++x) {
			const float* s = diff + y * diffStride + 2 * x;
			float v = *(s++);
			int best_o = (int)(*s);

			// Add to 4 historgrams around pixel using bilinear interpolation
			float yp = (y + 0.5f) / sbin - 0.5f;
			float xp = (x + 0.5f) / sbin - 0.5f;
			int iyp = (int)std::floor(yp);
			int ixp = (int)std::floor(xp);
			float vy0 = yp - iyp;
			float vx0 = xp - ixp;
			float vy1 = 1.0 - vy0;
			float vx1 = 1.0 - vx0;

			// Fill the value into the 4 neighborhood cells
			float *base = hist + iyp * histStride + ixp * numOrient + best_o;
			if (iyp >= 0 && ixp >= 0)
				*base += vy1 * vx1 * v;
			base += numOrient;
			if (iyp >= 0 && ixp + 1 < blockSize.x)
				*base += vx0 * vy1 * v;
			base += histStride;
			if (iyp + 1 < blockSize.y && ixp + 1 < blockSize.x)
				*base += vy0 * vx0 * v;
			base -= numOrient;
			if (iyp + 1 < blockSize.y && ixp >= 0)
				*base += vy0 * vx1 * v;
		}
	}

	// Compute the energy in each block by summing over orientation.
	ParallelFor([&](int64_t y) {
		float* src = hist + y * histStride;
		float* dst = norm + y * normStride;
		for (int x = 0; x < blockSize.x; ++x) {
			for (int o = 0; o < numOrient / 2; ++o) {
				float temp = *src + *(src + numOrient / 2);
				*dst += temp * temp;
				++src;
			}
			dst++;
			src += numOrient / 2;
		}
	}, blockSize.y, 16);

	// Compute the features.
	ParallelFor([&](int64_t y) {
		y += pady;
		float *src = hist + (y - pady + 1)*histStride + numOrient;
		float *dest[dimHOG];
		for (int i = 0; i < dimHOG; ++i) dest[i] = features[i].Data() + y * featStride + padx;
		float *p = norm + (y - pady)*normStride;
		float n1, n2, n3, n4;
		for (int x = padx; x < outSize.x - padx; ++x) {
			n4 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
			p += normStride;
			n3 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
			p += 1;
			n1 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
			p -= normStride;
			n2 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);

			float t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

			// Contrast-sesitive features
			for (int o = 0; o < numOrient; ++o) {
				float val = *src;
				float h1 = std::min(val*n1, 0.2f);
				float h2 = std::min(val*n2, 0.2f);
				float h3 = std::min(val*n3, 0.2f);
				float h4 = std::min(val*n4, 0.2f);
				*(dest[o]++) = 0.5 * (h1 + h2 + h3 + h4);
				++src;
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
			}

#if HOG_FEATURE_COUNT > 18
			// Contrast-insensitive features
			src -= numOrient;
			for (int o = 0; o < numOrient / 2; ++o) {
				float sum = *src + *(src + numOrient / 2);
				float h1 = std::min(sum*n1, 0.2f);
				float h2 = std::min(sum*n2, 0.2f);
				float h3 = std::min(sum*n3, 0.2f);
				float h4 = std::min(sum*n4, 0.2f);
				*(dest[numOrient + o]++) = 0.5 * (h1 + h2 + h3 + h4);
				++src;
			}
			src += numOrient / 2;

#endif

#if HOG_FEATURE_COUNT > 27
			// Texture features
			*(dest[27]++) = 0.2357f * t1;
			*(dest[28]++) = 0.2357f * t2;
			*(dest[29]++) = 0.2357f * t3;
			*(dest[30]++) = 0.2357f * t4;
#endif

#if HOG_FEATURE_COUNT > 31
			// Truncation feature
			*(dest[31]++) = 0.0f;
#endif
		}
	}, outSize.y - 2 * pady, 16);

#if HOG_FEATURE_COUNT > 31
	// Truncation features.
	int bt = pady - 1, bb = outSize.y - pady, bl = padx - 1, br = outSize.x - padx;
	float *featTrun = features[31].Data();
	ParallelFor([&](int64_t m) {
		for (int n = 0; n < outSize.x; ++n) {
			if (m > bt && m < bb && n > bl && n < br)
				continue;
			*(featTrun + m * featStride + n) = 1.0f;
		}
	}, outSize.y, 16);
#endif

	GFeatArenas[ThreadIndex].Reset();
}

//
// CN Features
//

void FeaturesExtractor::GetFeaturesCN(const Mat &img, std::vector<MatF> &features, const Vector2i &OutSize, MemoryArena *targetArena) {
	if (img.Channels() != 3)
		Critical("FeaturesExtractor::GetFeaturesCN: cannot get the CN features on current image.");
	if (OutSize.x <= 0 || OutSize.y <= 0)
		Critical("FeaturesExtractor::GetFeaturesCN: must specify the output size.");

	Vector2i imSize(img.Cols(), img.Rows());
	std::vector<MatF> tempMats(10, &GFeatArenas[ThreadIndex]);
	for (int i = 0; i < 10; ++i)
		tempMats[i].Reshape(img.Rows(), img.Cols());
	const size_t imStride = img.Cols() * 3;
	const size_t featStride = img.Cols();
	const uint8_t *im = img.Data();

	ParallelFor([&](int64_t y) {
		const uint8_t *src = im + y * imStride;
		float *dest[10];
		for (int i = 0; i < 10; ++i) dest[i] = tempMats[i].Data() + featStride * y;
		for (int x = 0; x < imSize.x; ++x) {
			int index = (int)(std::floor((float)src[0] / 8) + 32 * std::floor((float)src[1] / 8) + 32 * 32 * std::floor((float)src[2] / 8));
			src += 3;
			for (int k = 0; k < 10; k++)
				*(dest[k]++) = ColorNames[index][k];
		}
	}, imSize.y, 32);

	features.clear();
	features.resize(10, MatF(targetArena));
	ParallelFor([&](uint64_t index) {
		tempMats[index].Resize(features[index], OutSize.y, OutSize.x);
	}, 10, 1);

	GFeatArenas[ThreadIndex].Reset();
}

//
// RGB Features
//

void FeaturesExtractor::GetFeaturesRGB(const Mat& img, std::vector<MatF>& features, const Vector2i& OutSize, MemoryArena *targetArena) {
	if (img.Channels() != 3)
		Critical("FeaturesExtractor::GetFeaturesRGB: cannot get the RGB features on current image.");
	if (OutSize.x <= 0 || OutSize.y <= 0)
		Critical("FeaturesExtractor::GetFeaturesRGB: must specify the output size.");

	std::vector<MatF> tempMats(3, &GFeatArenas[ThreadIndex]);
	for (int i = 0; i < 3; ++i) tempMats[i].Reshape(img.Rows(), img.Cols());
	float means[3] = { 0.0f, 0.0f, 0.0f };
	int pureCount = tempMats[0].Size();
	int bc = MaxThreadIndex();
	int bs = pureCount / bc;
	std::mutex lock;

	ParallelFor([&](int64_t t) {
		int start = t * bs;
		int end = std::min(start + bs, pureCount);
		float sumt[3] = { 0.0f, 0.0f, 0.0f };
		const uint8_t *src = img.Data() + start * 3;
		float *dest[3] = {
			tempMats[0].Data() + start,
			tempMats[1].Data() + start,
			tempMats[2].Data() + start
		};

		for (int i = start; i < end; ++i) {
			float temp = (float)(*(src++)) / 255.0f - 0.5f;
			*(dest[0]++) = temp;
			sumt[0] += temp;
			temp = (float)(*(src++)) / 255.0f - 0.5f;
			*(dest[1]++) = temp;
			sumt[1] += temp;
			temp = (float)(*(src++)) / 255.0f - 0.5f;
			*(dest[2]++) = temp;
			sumt[2] += temp;
		}

		lock.lock();
		means[0] += sumt[0];
		means[1] += sumt[1];
		means[2] += sumt[2];
		lock.unlock();
	}, bc, 1);

	ParallelFor([&](int64_t i) {
		means[i] /= pureCount;
		tempMats[i].ReValue(-means[i]);
	}, 3, 1);

	features.clear();
	features.resize(3, MatF(targetArena));
	ParallelFor([&](uint64_t index) {
		tempMats[index].Resize(features[index], OutSize.y, OutSize.x);
	}, 3, 1);

	GFeatArenas[ThreadIndex].Reset();
}

}	// namespace CSRT