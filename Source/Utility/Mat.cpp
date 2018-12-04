//
// Baic image operations.
//

#include "Mat.h"
#include "Parallel.h"
#include "Memory.h"
#include <unordered_map>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace CSRT {

// Image loading arena. Manually reseted.
static MemoryArena GImgLoadArena;

//Image resize arena. Each threads acces it own arena. Automatically reseted.
static MemoryArena *GImgResizeArenas = nullptr;
static MemoryArena *GImgResizeArenasL2 = nullptr;

std::unordered_map<std::string, void*> HResizer;
std::mutex HMutex;

//
// ImageMemoryArena Implementation
//

ImageMemoryArena GImageMemoryArena;

bool ImageMemoryArena::SingleInstance = false;

ImageMemoryArena::ImageMemoryArena() : initialized(false) {
	if (SingleInstance) {
		Critical("Only one \"ImageMemoryArena\" can exists.");
		return;
	}
	SingleInstance = true;
}

ImageMemoryArena::~ImageMemoryArena() {
	HResizer.clear();
	if (GImgResizeArenas) {
		delete[] GImgResizeArenas;
		GImgResizeArenas = nullptr;
	}
    if (GImgResizeArenasL2) {
        delete[] GImgResizeArenasL2;
        GImgResizeArenasL2 = nullptr;
    }
	SingleInstance = false;
}

void ImageMemoryArena::Initialize() {
	if (initialized) return;
	initialized = true;
	HResizer.clear();
	GImgResizeArenas = new MemoryArena[MaxThreadIndex()];
    GImgResizeArenasL2 = new MemoryArena[MaxThreadIndex()];
}

void ImageMemoryArena::ResetImgLoadArena() {
	GImgLoadArena.Reset();
}

#ifdef WITH_OPENCV
cv::Mat cvImgReadM;
#endif

}	// namespace CSRT

	// Stb image loading. Note, use arena for image loading. For data persistence, please copy the loaded data to a safe place.
	// Please manually reset the arena after the image load cycle.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_MALLOC(sz)						CSRT::GImgLoadArena.Alloc(size)
#define STBI_REALLOC(p,newsz)				CSRT::Critical("Stb realloc not supported.")
#define STBI_REALLOC_SIZED(p,oldsz,newsz)	CSRT::GImgLoadArena.Realloc(p,oldsz,newsz)
#define STBI_FREE(p)						((void)p)
#include "../../ThirdParty/Stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#undef STBI_ONLY_JPEG
#undef STBI_MALLOC
#undef STBI_REALLOC
#undef STBI_REALLOC_SIZED
#undef STBI_FREE

// Stb image writing. Use default malloc and free routine. Used only for debug purpose.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(_MSC_VER)
#define STBI_MSC_SECURE_CRT
#endif
#include "../../ThirdParty/Stb/stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(_MSC_VER)
#undef STBI_MSC_SECURE_CRT
#endif


namespace CSRT {

using namespace std;

//
// Custom Bilinear Resize implementation.
//

const int LINEAR_SHIFT = 4;
const int LINEAR_ROUND_TERM = 1 << (LINEAR_SHIFT - 1);
const int BILINEAR_SHIFT = LINEAR_SHIFT * 2;
const int BILINEAR_ROUND_TERM = 1 << (BILINEAR_SHIFT - 1);
const int FRACTION_RANGE = 1 << LINEAR_SHIFT;
const double FRACTION_ROUND_TERM = 0.5 / FRACTION_RANGE;

template<typename T>
struct Buffer {
	Buffer(int width) : ix(nullptr), ax(nullptr), iy(nullptr), ay(nullptr) {
		pT = GImgResizeArenasL2[ThreadIndex].Alloc<T>(width * 2, false);
		pbx[0] = pT;
		pbx[1] = pT + width;
	}

	~Buffer() {
		GImgResizeArenasL2[ThreadIndex].Reset();
	}

	int *ix;
	float *ax;
	int *iy;
	float *ay;
	T *pbx[2];
private:
	T *pT;
};

template<typename T>
struct BufferNearest {
    BufferNearest(int width) : ix(nullptr), ax(nullptr), iy(nullptr), ay(nullptr) {
        pT = GImgResizeArenasL2[ThreadIndex].Alloc<T>(width * 2, false);
        pbx[0] = pT;
        pbx[1] = pT + width;
    }

    ~BufferNearest() {
        GImgResizeArenasL2[ThreadIndex].Reset();
    }

    int *ix;
    float *ax;
    int *iy;
    float *ay;
    T *pbx[2];
private:
    T *pT;
};

void EstimateAlphaIndex(int srcSize, int dstSize, int *indexes, float *alphas, int channelCount) {
	float scale = (float)srcSize / dstSize;
	for (int i = 0; i < dstSize; ++i) {
		float alpha = (i + 0.5f)*scale - 0.5f;
		int index = (int)std::floor(alpha);
		alpha -= index;

		if (index < 0) {
			index = 0;
			alpha = 0.0f;
		}
		if (index > srcSize - 2) {
			index = srcSize - 2;
			alpha = 1.0f;
		}

		for (int c = 0; c < channelCount; c++) {
			int offset = i * channelCount + c;
			indexes[offset] = channelCount * index + c;
			alphas[offset] = alpha;
		}
	}
}

template<typename T>
void ResizeBilinear(
	const T *src, int srcWidth, int srcHeight, int srcStride,
	T *dst, int dstWidth, int dstHeight, int dstStride, int channelCount) {
	int dstRowSize = channelCount * dstWidth;
	Buffer<T> buffer(dstRowSize);

	std::string key0 = StringPrintf("f%ds%dd%dc", srcHeight, dstHeight, 1);
	std::string key1 = StringPrintf("f%ds%dd%dc", srcWidth, dstWidth, channelCount);
	HMutex.lock();
	if(HResizer.find(key0) != HResizer.end()) {
		buffer.iy = reinterpret_cast<int*>(HResizer[key0]);
		buffer.ay = reinterpret_cast<float*>(buffer.iy + dstHeight);
	} else {
		HResizer[key0] = reinterpret_cast<void*>(GImgResizeArenas[ThreadIndex].Alloc(dstHeight * (sizeof(int) + sizeof(float))));
		buffer.iy = reinterpret_cast<int*>(HResizer[key0]);
		buffer.ay = reinterpret_cast<float*>(buffer.iy + dstHeight);
		EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);
	}
	if(HResizer.find(key1) != HResizer.end()) {
		buffer.ix = reinterpret_cast<int*>(HResizer[key1]);
		buffer.ax = reinterpret_cast<float*>(buffer.ix + dstRowSize);
	} else {
		HResizer[key1] = reinterpret_cast<void*>(GImgResizeArenas[ThreadIndex].Alloc(dstRowSize * (sizeof(int) + sizeof(float))));
		buffer.ix = reinterpret_cast<int*>(HResizer[key1]);
		buffer.ax = reinterpret_cast<float*>(buffer.ix + dstRowSize);
		EstimateAlphaIndex(srcWidth, dstWidth, buffer.ix, buffer.ax, channelCount);
	}
	HMutex.unlock();

	int previous = -2;
	for (int yDst = 0; yDst < dstHeight; yDst++, dst += dstStride) {
		float fy = buffer.ay[yDst];
		int sy = buffer.iy[yDst];
		int k = 0;

		if (sy == previous)
			k = 2;
		else if (sy == previous + 1) {
			std::swap(buffer.pbx[0], buffer.pbx[1]);
			k = 1;
		}
		previous = sy;

		for (; k < 2; k++) {
			T *pb = buffer.pbx[k];
			const T* ps = src + (sy + k)*srcStride;
			for (int x = 0; x < dstRowSize; x++) {
				int sx = buffer.ix[x];
				float fx = buffer.ax[x];
				T t = ps[sx];
				pb[x] = t + (ps[sx + channelCount] - t)*fx;
			}
		}

		if (fy == 0)
			memcpy(dst, buffer.pbx[0], dstRowSize * sizeof(T));
		else if (fy == 1.0f)
			memcpy(dst, buffer.pbx[1], dstRowSize * sizeof(T));
		else {
			for (int xDst = 0; xDst < dstRowSize; xDst++) {
				T t = buffer.pbx[0][xDst];
				dst[xDst] = t + (buffer.pbx[1][xDst] - t)*fy;
			}
		}
	}
}

template<typename T>
void ResizeNearest(
        const T *src, int srcWidth, int srcHeight, int srcStride,
        T *dst, int dstWidth, int dstHeight, int dstStride, int channelCount) {
    int dstRowSize = channelCount * dstWidth;
    BufferNearest<T> buffer(dstRowSize);

    std::string key0 = StringPrintf("f%ds%dd%dc", srcHeight, dstHeight, 1);
    std::string key1 = StringPrintf("f%ds%dd%dc", srcWidth, dstWidth, channelCount);
    HMutex.lock();
    if(HResizer.find(key0) != HResizer.end()) {
        buffer.iy = reinterpret_cast<int*>(HResizer[key0]);
        buffer.ay = reinterpret_cast<float*>(buffer.iy + dstHeight);
    } else {
        HResizer[key0] = reinterpret_cast<void*>(GImgResizeArenas[ThreadIndex].Alloc(dstHeight * (sizeof(int) + sizeof(float))));
        buffer.iy = reinterpret_cast<int*>(HResizer[key0]);
        buffer.ay = reinterpret_cast<float*>(buffer.iy + dstHeight);
        EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);
    }
    if(HResizer.find(key1) != HResizer.end()) {
        buffer.ix = reinterpret_cast<int*>(HResizer[key1]);
        buffer.ax = reinterpret_cast<float*>(buffer.ix + dstRowSize);
    } else {
        HResizer[key1] = reinterpret_cast<void*>(GImgResizeArenas[ThreadIndex].Alloc(dstRowSize * (sizeof(int) + sizeof(float))));
        buffer.ix = reinterpret_cast<int*>(HResizer[key1]);
        buffer.ax = reinterpret_cast<float*>(buffer.ix + dstRowSize);
        EstimateAlphaIndex(srcWidth, dstWidth, buffer.ix, buffer.ax, channelCount);
    }
    HMutex.unlock();

    int previous = -2;
    for (int yDst = 0; yDst < dstHeight; yDst++, dst += dstStride) {
        float fy = buffer.ay[yDst];
        int sy = buffer.iy[yDst];
        int k = 0;

        if (sy == previous)
            k = 2;
        else if (sy == previous + 1) {
            std::swap(buffer.pbx[0], buffer.pbx[1]);
            k = 1;
        }
        previous = sy;

        for (; k < 2; k++) {
            T *pb = buffer.pbx[k];
            const T* ps = src + (sy + k)*srcStride;
            for (int x = 0; x < dstRowSize; x++) {
                int sx = buffer.ix[x];
                float fx = buffer.ax[x];
                pb[x] = fx > 0.5f ? ps[sx + channelCount] : ps[sx];
            }
        }

        if (fy == 0)
            memcpy(dst, buffer.pbx[0], dstRowSize * sizeof(T));
        else if (fy == 1.0f)
            memcpy(dst, buffer.pbx[1], dstRowSize * sizeof(T));
        else {
            for (int xDst = 0; xDst < dstRowSize; xDst++) {
                dst[xDst] = fy > 0.5f ? buffer.pbx[1][xDst] : buffer.pbx[0][xDst];
            }
        }
    }
}

template<>
struct Buffer<uint8_t> {
	Buffer(int width) : ix(nullptr), ax(nullptr), iy(nullptr), ay(nullptr) {
		_p = GImgResizeArenasL2[ThreadIndex].Alloc<int>(2 * width, false);
		pbx[0] = _p;
		pbx[1] = pbx[0] + width;
	}

	~Buffer() {
		GImgResizeArenasL2[ThreadIndex].Reset();
	}

	int *ix;
	int *ax;
	int *iy;
	int *ay;
	int *pbx[2];
private:
	int *_p;
};

void EstimateAlphaIndex_uint8_t(int srcSize, int dstSize, int *indexes, int *alphas, int channelCount) {
	float scale = (float)srcSize / dstSize;
	for (int i = 0; i < dstSize; ++i) {
		float alpha = (float)((i + 0.5f)*scale - 0.5f);
		int index = (int)std::floor(alpha);
		alpha -= index;

		if (index < 0) {
			index = 0;
			alpha = 0;
		}
		if (index > srcSize - 2) {
			index = srcSize - 2;
			alpha = 1;
		}

		for (int c = 0; c < channelCount; c++) {
			int offset = i * channelCount + c;
			indexes[offset] = (int)(channelCount*index + c);
			alphas[offset] = (int)(alpha * FRACTION_RANGE + 0.5f);
		}
	}
}

template<>
void ResizeBilinear<uint8_t>(
	const uint8_t *src, int srcWidth, int srcHeight, int srcStride,
	uint8_t *dst, int dstWidth, int dstHeight, int dstStride, int channelCount) {
	int dstRowSize = channelCount * dstWidth;
	Buffer<uint8_t> buffer(dstRowSize);

	std::string key0 = StringPrintf("i%ds%dd%dc", srcHeight, dstHeight, 1);
	std::string key1 = StringPrintf("i%ds%dd%dc", srcWidth, dstWidth, channelCount);
	HMutex.lock();
	if(HResizer.find(key0) != HResizer.end()) {
		buffer.iy = reinterpret_cast<int*>(HResizer[key0]);
		buffer.ay = buffer.iy + dstHeight;
	} else {
		HResizer[key0] = reinterpret_cast<void*>(GImgResizeArenas[ThreadIndex].Alloc(dstHeight * 2 * sizeof(int)));
		buffer.iy = reinterpret_cast<int*>(HResizer[key0]);
		buffer.ay = buffer.iy + dstHeight;
		EstimateAlphaIndex_uint8_t(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);
	}
	if(HResizer.find(key1) != HResizer.end()) {
		buffer.ix = reinterpret_cast<int*>(HResizer[key1]);
		buffer.ax = buffer.ix + dstRowSize;
	} else {
		HResizer[key1] = reinterpret_cast<void*>(GImgResizeArenas[ThreadIndex].Alloc(dstRowSize * 2 * sizeof(int)));
		buffer.ix = reinterpret_cast<int*>(HResizer[key1]);
		buffer.ax = buffer.ix + dstRowSize;
		EstimateAlphaIndex_uint8_t(srcWidth, dstWidth, buffer.ix, buffer.ax, channelCount);
	}
	HMutex.unlock();

	int previous = -2;
	for (int yDst = 0; yDst < dstHeight; yDst++, dst += dstStride) {
		int fy = buffer.ay[yDst];
		int sy = buffer.iy[yDst];
		int k = 0;

		if (sy == previous)
			k = 2;
		else if (sy == previous + 1) {
			std::swap(buffer.pbx[0], buffer.pbx[1]);
			k = 1;
		}
		previous = sy;

		for (; k < 2; k++) {
			int* pb = buffer.pbx[k];
			const uint8_t* ps = src + (sy + k)*srcStride;
			for (int x = 0; x < dstRowSize; x++) {
				int sx = buffer.ix[x];
				int fx = buffer.ax[x];
				int t = ps[sx];
				pb[x] = (t << LINEAR_SHIFT) + (ps[sx + channelCount] - t)*fx;
			}
		}

		if (fy == 0)
			for (int xDst = 0; xDst < dstRowSize; xDst++)
				dst[xDst] = ((buffer.pbx[0][xDst] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
		else if (fy == FRACTION_RANGE)
			for (int xDst = 0; xDst < dstRowSize; xDst++)
				dst[xDst] = ((buffer.pbx[1][xDst] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
		else {
			for (int xDst = 0; xDst < dstRowSize; xDst++) {
				int t = buffer.pbx[0][xDst];
				dst[xDst] = ((t << LINEAR_SHIFT) + (buffer.pbx[1][xDst] - t)*fy + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
			}
		}
	}
}

//
// Mat Implementation.
//

Mat::Mat(int rowNum, int colNum, int channelNum, MemoryArena *storage)
	: rows(rowNum), cols(colNum), channels(channelNum), arena(storage), data(nullptr) {
	size = rows * cols * channels;
	if (arena) data = arena->Alloc<uint8_t>(size, false);
	else data = AllocAligned<uint8_t>(size);
	memset(data, 0, size);
}

Mat::Mat(const std::string& filePath)
	: rows(0), cols(0), channels(0), arena(&GImgLoadArena), data(nullptr) {
#ifdef WITH_OPENCV
    cvImgReadM = cv::imread(filePath);
	rows = cvImgReadM.rows;
	cols = cvImgReadM.cols;
	channels = cvImgReadM.channels();
	size = rows * cols * channels;
	data = GImgLoadArena.Alloc<uint8_t>(size);
	memcpy(data, cvImgReadM.data, size * sizeof(uint8_t));
#else
	data = stbi_load(filePath.c_str(), &cols, &rows, &channels, 0);
	size = rows * cols * channels;
	if (data == nullptr)
		Critical("Mat load failed for file: " + filePath + stbi_failure_reason());
#endif
}

Mat::Mat(const uint8_t *sourceData, int rowNum, int colNum, int channelNum, MemoryArena *storage)
	: rows(rowNum), cols(colNum), channels(channelNum), arena(storage), data(nullptr) {
	size = rows * cols * channels;
	if (arena) data = arena->Alloc<uint8_t>(size, false);
	else data = AllocAligned<uint8_t>(size);
	memcpy(data, sourceData, size);
}

Mat::Mat(const Mat &m)
	: rows(m.rows), cols(m.cols), channels(m.channels), size(m.size), arena(m.arena) {
	if (arena) data = arena->Alloc<uint8_t>(size, false);
	else data = AllocAligned<uint8_t>(size);
	memcpy(data, m.data, size);
}

Mat::Mat(Mat &&m) noexcept
	: rows(m.rows), cols(m.cols), channels(m.channels), size(m.size), arena(m.arena), data(m.data) {
	m.data = nullptr;
	m.rows = m.cols = m.channels = m.size = 0;
	m.arena = nullptr;
}

Mat &Mat::operator=(const Mat &m) {
	rows = m.rows;
	cols = m.cols;
	channels = m.channels;
	size = m.size;
	if (data && !arena)
		FreeAligned(data);
	if (arena) data = arena->Alloc<uint8_t>(size, false);
	else data = AllocAligned<uint8_t>(size);
	memcpy(data, m.data, size);
	return *this;
}

Mat &Mat::operator=(Mat &&m) noexcept {
	rows = m.rows;
	cols = m.cols;
	channels = m.channels;
	size = m.size;
	if (data && !arena)
		FreeAligned(data);
	if (arena == m.arena) {
		data = m.data;
		m.data = nullptr;
		m.arena = nullptr;
		m.rows = m.cols = m.channels = m.size = 0;
	} else {
		if (arena) data = arena->Alloc<uint8_t>(size, false);
		else data = AllocAligned<uint8_t>(size);
		memcpy(data, m.data, size);
	}
	return *this;
}

Mat &Mat::Reshape(int rowNum, int colNum, int channelNum, bool zero) {
	if (rowNum == rows && colNum == cols && channelNum == channels) {
		if (zero)
			memset(data, 0, size);
		return *this;
	}

	rows = rowNum;
	cols = colNum;
	channels = channelNum;
	size = rows * cols * channels;
	if (data && !arena)
		FreeAligned(data);
	if (arena) data = arena->Alloc<uint8_t>(size, false);
	else data = AllocAligned<uint8_t>(size);

	if (zero)
		memset(data, 0, size);
	return *this;
}

Mat::~Mat() {
	if (data && !arena)
		FreeAligned(data);
}

void Mat::ManageExternalData(uint8_t *sourceData, int rowNum, int colNum, int channelNum) {
    if (data && !arena)
        FreeAligned(data);
    data = sourceData;
    arena = &GImgLoadArena;
    rows = rowNum;
    cols = colNum;
    channels = channelNum;
    size = rows * cols * channels;
}

float Mat::Mean() const {
	float sum = 0.0f;
	const uint8_t *ptemp = data;
	for (int i = 0; i < size; ++i)
		sum += *(ptemp++);
	return sum / size;
}

void Mat::ReValue(float offset) {
	uint8_t *ptemp = data;
	for (int i = 0; i < size; ++i) {
		*ptemp = (uint8_t)Clamp((int)(*ptemp + offset), 0, 255);
		++ptemp;
	}
}

void Mat::ReValue(float scale, float offset) {
	uint8_t *ptemp = data;
	for (int i = 0; i < size; ++i) {
		*ptemp = (uint8_t)Clamp((int)(*ptemp * scale + offset), 0, 255);
		++ptemp;
	}
}

void Mat::MinMaxLoc(uint8_t& min, uint8_t& max, Vector2i& minLoc, Vector2i& maxLoc) const {
	if (size <= 0)
		Critical("Mat::MinMaxLoc: cannot evalute target value on empty mat.");

	min = max = data[0];
	int minIdx = 0, maxIdx = 0;
	const uint8_t *ptemp = data + 1;
	for (int i = 1; i < size; ++i) {
		if (*ptemp < min) {
			min = *ptemp;
			minIdx = i;
		} else if (*ptemp > max) {
			max = *ptemp;
			maxIdx = i;
		}
		++ptemp;
	}
	int stride = cols * channels;
	minLoc.y = minIdx / stride;
	minLoc.x = minIdx - minLoc.y * stride;
	maxLoc.y = maxIdx / stride;
	maxLoc.x = maxIdx - maxLoc.y * stride;
}

void Mat::MinLoc(uint8_t &min, Vector2i &minLoc) const {
	if (size <= 0)
		Critical("Mat::MinLoc: cannot evalute target value on empty mat.");

	min = data[0];
	int minIdx = 0;
	const uint8_t *ptemp = data + 1;
	for (int i = 1; i < size; ++i) {
		if (*ptemp < min) {
			min = *ptemp;
			minIdx = i;
		}
		++ptemp;
	}
	int stride = cols * channels;
	minLoc.y = minIdx / stride;
	minLoc.x = minIdx - minLoc.y * stride;
}

void Mat::MaxLoc(uint8_t &max, Vector2i &maxLoc) const {
	if (size <= 0)
		Critical("Mat::MaxLoc: cannot evalute target value on empty mat.");

	max = data[0];
	int maxIdx = 0;
	const uint8_t *ptemp = data + 1;
	for (int i = 1; i < size; ++i) {
		if (*ptemp > max) {
			max = *ptemp;
			maxIdx = i;
		}
		++ptemp;
	}
	int stride = cols * channels;
	maxLoc.y = maxIdx / stride;
	maxLoc.x = maxIdx - maxLoc.y * stride;
}

void Mat::Split(std::vector<Mat> &mats, MemoryArena *targetArena) const {
	mats.clear();
	mats.resize(channels, Mat(targetArena));
	for (int i = 0; i < channels; ++i)
		mats[i].Reshape(rows, cols, 1, false);
	const uint8_t *pt = data;
	// Allocate in stack, automatically freed.
	uint8_t **pts = ALLOCA(uint8_t*, channels);
	for (int k = 0; k < channels; ++k)
		pts[k] = mats[k].data;

	int tarlen = size / channels;
	for (int index = 0; index < tarlen; ++index) {
		for (int k = 0; k < channels; ++k)
			*(pts[k]++) = *(pt++);
	}
}

void Mat::Merge(const std::vector<Mat> &mats) {
	this->Reshape(mats[0].rows, mats[0].cols, (int)mats.size(), false);
	uint8_t *pt = data;
	// Allocate in stack, automatically freed.
	uint8_t **pts = ALLOCA(uint8_t*, channels);
	for (int k = 0; k < channels; ++k)
		pts[k] = mats[k].data;

	int tarlen = size / channels;
	for (int index = 0; index < tarlen; ++index) {
		for (int k = 0; k < channels; ++k)
			*(pt++) = *(pts[k]++);
	}
}

void Mat::Resize(Mat &dest, float sr, float sc, ResizeMode mode) const {
	int trows = (int)(rows * sr);
	int tcols = (int)(cols * sc);
	dest.Reshape(trows, tcols, channels, false);
#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_8UC(channels), data);
	cv::Mat cvDstM(trows, tcols, CV_8UC(channels), dest.data);
	int interp = cv::INTER_LINEAR;
	switch (mode) {
		case ResizeMode::Bicubic:
			interp = cv::INTER_CUBIC;
			break;
		case ResizeMode::Nearest:
			interp = cv::INTER_NEAREST;
			break;
		default:
			interp = cv::INTER_LINEAR;
			break;
	}
	cv::resize(cvOrgM, cvDstM, cv::Size(tcols, trows), 0, 0, interp);
#else
	if(mode == ResizeMode::Nearest) {
		ResizeNearest(data, cols, rows, cols * channels,
		  dest.data, dest.cols, dest.rows, dest.cols * dest.channels, channels);
	} else {
		ResizeBilinear(data, cols, rows, cols * channels,
		   dest.data, dest.cols, dest.rows, dest.cols * dest.channels, channels);
	}
#endif
}

void Mat::Resize(Mat &dest, int tr, int tc, ResizeMode mode) const {
	dest.Reshape(tr, tc, channels, false);
#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_8UC(channels), data);
	cv::Mat cvDstM(tr, tc, CV_8UC(channels), dest.data);
	int interp = cv::INTER_LINEAR;
	switch (mode) {
		case ResizeMode::Bicubic:
			interp = cv::INTER_CUBIC;
			break;
		case ResizeMode::Nearest:
			interp = cv::INTER_NEAREST;
			break;
		default:
			interp = cv::INTER_LINEAR;
			break;
	}
	cv::resize(cvOrgM, cvDstM, cv::Size(tc, tr), 0, 0, interp);
#else
	if(mode == ResizeMode::Nearest) {
		ResizeNearest(data, cols, rows, cols * channels,
					  dest.data, dest.cols, dest.rows, dest.cols * dest.channels, channels);
	} else {
		ResizeBilinear(data, cols, rows, cols * channels,
					   dest.data, dest.cols, dest.rows, dest.cols * dest.channels, channels);
	};
#endif
}

void Mat::RGBToHSV(Mat &dest) const {
	if (channels != 3)
		Critical("Mat::RGBToHSV: only 3 channels rgb image data can be converted to hsv.");
	dest.Reshape(rows, cols, 3);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_8UC3, data);
	cv::Mat cvDstM(rows, cols, CV_8UC3, dest.data);
	cv::cvtColor(cvOrgM, cvDstM, cv::COLOR_BGR2HSV);
	uint8_t *pdest = dest.data;
	for(int i = 0; i < dest.size; i += 3) {
	    pdest[i] *= Clamp(pdest[i] * (255.0f / 180.0f), 0, 255);
	}
#else
	const uint8_t *source = data;
	uint8_t *destination = dest.data;
	ParallelFor([&](int64_t i) {
		const uint8_t *ps = source + i * cols * 3;
		uint8_t *pd = destination + i * cols * 3;
		for (int j = 0; j < cols; ++j) {
			float r = *(ps++) / 255.0f;
			float g = *(ps++) / 255.0f;
			float b = *(ps++) / 255.0f;
			float cmax = std::max(r, std::max(g, b));
			float cmin = std::min(r, std::min(g, b));
			float delta = cmax - cmin;

			// Black-gray-white
			if (cmax == cmin) {
				pd += 2;
				*(pd++) = (uint8_t)Clamp((int)(cmax * 255.0f), 0, 255);
				continue;
			}

			// Hue
			float hue = 0.0f;
			if (r == cmax)
				hue = (g - b) / delta;
			else if (g == cmax)
				hue = (b - r) / delta + 2.0f;
			else
				hue = (r - g) / delta + 4.0f;
			hue *= 60.0f;
			if (hue < 0.0f) hue += 360.0f;
			*(pd++) = (uint8_t)Clamp((int)(hue / 360.0f * 255.0f), 0, 255);

			// Saturation
			*(pd++) = (uint8_t)Clamp((int)(delta / cmax * 255.0f), 0, 255);

			// Value
			*(pd++) = (uint8_t)Clamp((int)(cmax * 255.0f), 0, 255);
		}
	}, rows, 32);
#endif
}

void Mat::HSVToRGB(Mat &dest) const {
	if (channels != 3)
		Critical("Mat::HSVToRGB: only 3 channels hsv image data can be converted to rgb.");
	dest.Reshape(rows, cols, 3);

#ifdef WITH_OPENCV
    uint8_t *pdest = data;
    for(int i = 0; i < size; i += 3) {
        pdest[i] *= Clamp(pdest[i] * (180.0f / 255.0f), 0, 255);
    }
    cv::Mat cvOrgM(rows, cols, CV_8UC3, data);
    cv::Mat cvDstM(rows, cols, CV_8UC3, dest.data);
    cv::cvtColor(cvOrgM, cvDstM, cv::COLOR_HSV2BGR);
#else
    const uint8_t *source = data;
	uint8_t *destination = dest.data;
	ParallelFor([&](int64_t i) {
		const uint8_t *ps = source + i * cols * 3;
		uint8_t *pd = destination + i * cols * 3;
		for (int j = 0; j < cols; ++j) {
			float h = *(ps++) / 255.0f * 360.f / 60.0f;
			float s = *(ps++) / 255.0f;
			if (s <= 0.0f) {
				*(pd++) = *ps;
				*(pd++) = *ps;
				*(pd++) = *ps;
				++ps;
				continue;
			}
			float v = *(ps++) / 255.0f;

			int hi = (int)h;
			float hf = h - hi;
			float p = v * (1.0f - s);
			float q = v * (1.0f - (s * hf));
			float t = v * (1.0f - (s * (1.0f - hf)));

			float r, g, b;
			switch (hi) {
			case 0:
				r = v; g = t; b = p; break;
			case 1:
				r = q; g = v; b = p; break;
			case 2:
				r = p; g = v; b = t; break;
			case 3:
				r = p; g = q; b = v; break;
			case 4:
				r = t; g = p; b = v; break;
			case 5:
			default:
				r = v; g = p; b = q; break;
			}

			*(pd++) = (uint8_t)Clamp((int)(r * 255.0f), 0, 255);
			*(pd++) = (uint8_t)Clamp((int)(g * 255.0f), 0, 255);
			*(pd++) = (uint8_t)Clamp((int)(b * 255.0f), 0, 255);
		}
	}, rows, 32);
#endif
}

void Mat::RGBToGray(Mat &dest) const {
	if (channels != 3)
		Critical("Mat::RGBToGray: only 3 channels rgb image data can be converted to gray.");
	dest.Reshape(rows, cols, 1);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_8UC3, data);
	cv::Mat cvDstM(rows, cols, CV_8UC1, dest.data);
	cv::cvtColor(cvOrgM, cvDstM, cv::COLOR_BGR2GRAY);
#else
	const uint8_t *source = data;
	uint8_t *destination = dest.data;
	ParallelFor([&](int64_t i) {
		const uint8_t *ps = source + i * cols * 3;
		uint8_t *pd = destination + i * cols;
		for (int j = 0; j < cols; ++j) {
			uint8_t r, g, b;
			r = *(ps++);
			g = *(ps++);
			b = *(ps++);
			uint8_t v = (uint8_t)Clamp((int)(0.299f * r + 0.587f * g + 0.114f * b), 0, 255);
			*(pd++) = v;
		}
	}, rows, 32);
#endif
}

void Mat::GrayToRGB(Mat &dest) const {
	if (channels != 1)
		Critical("Mat::GrayToRGB: only 1 channel gray image data can be converted to rgb.");
	dest.Reshape(rows, cols, 3);

	const uint8_t *source = data;
	uint8_t *destination = dest.data;
	ParallelFor([&](int64_t i) {
		const uint8_t *ps = source + i * cols;
		uint8_t *pd = destination + i * cols * 3;
		for (int j = 0; j < cols; ++j) {
			uint8_t v = *(ps++);
			*(pd++) = v;
			*(pd++) = v;
			*(pd++) = v;
		}
	}, rows, 32);
}

void Mat::ToMatF(MatF &dest, float scale, float offset) const {
	dest.Reshape(rows, cols * channels, false);
	const uint8_t *ps = data;
	float *pd = dest.data;
	for (int i = 0; i < size; ++i)
		*(pd++) = *(ps++) * scale + offset;
}

void Mat::Roi(const Bounds2i& bound, Mat& outM) const {
	int trows = bound.pMax.y - bound.pMin.y;
	int tcols = bound.pMax.x - bound.pMin.x;
	outM.Reshape(trows, tcols, channels, false);
	int strideSrc = cols * channels;
	int strideDest = tcols * channels;

	for (int y = 0; y < trows; ++y) {
		const uint8_t *ps = data + (bound.pMin.y + y) * strideSrc + bound.pMin.x * channels;
		uint8_t *pd = outM.data + y * strideDest;
		memcpy(pd, ps, strideDest);
	}
}

bool Mat::MakeBorder(Mat& outM, int left, int top, int right, int bottom, bool replicate) const {
	if (left == 0 && top == 0 && right == 0 && bottom == 0)
		return false;

	int trows = rows + top + bottom;
	int tcols = cols + left + right;
	outM.Reshape(trows, tcols, channels);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_8UC(channels), data);
	cv::Mat cvDstM(trows, tcols, CV_8UC(channels), outM.data);
	cv::copyMakeBorder(cvOrgM, cvDstM, top, bottom, left, right, replicate ? cv::BORDER_REPLICATE : cv::BORDER_CONSTANT);
#else
	int strideSrc = cols * channels;
	int strideDest = tcols * channels;

	if (replicate) {
		for (int y = 0; y < trows; ++y) {
			uint8_t *pd = outM.data + y * strideDest + left * channels;
			const uint8_t *ps = nullptr;
			if (y < top)
				ps = data;
			else if (y > top + rows - 1)
				ps = data + (rows - 1) * strideSrc;
			else
				ps = data + (y - top) * strideSrc;

			memcpy(pd, ps, strideSrc);
			uint8_t *ls = outM.data + y * strideDest;
			uint8_t *rs = outM.data + y * strideDest + strideDest - right * channels;
			for (int i = 0; i < left; ++i) {
				for (int k = 0; k < channels; ++k)
					*(ls++) = ps[k];
			}
			ps += (strideSrc - channels);
			for (int i = 0; i < right; ++i) {
				for (int k = 0; k < channels; ++k)
					*(rs++) = ps[k];
			}
		}
	} else {
		for (int y = 0; y < rows; ++y) {
			const uint8_t *ps = data + y * strideSrc;
			uint8_t *pd = outM.data + (y + top) * strideDest + left * channels;
			memcpy(pd, ps, strideSrc);
		}
	}
#endif
	return true;
}

void Mat::Reduce(MatF& outM, ReduceMode mode, bool toRow) const {
	if (size == 0) {
		Critical("Mat::Reduce: cannot reduce an empty mat.");
		return;
	}
	int stride = cols * channels;

	if (toRow) {
		outM.Reshape(1, stride);
		float *pd = outM.data;
		switch (mode) {
		case ReduceMode::Sum:
			for (int x = 0; x < stride; ++x) {
				const uint8_t *ps = data + x;
				for (int y = 0; y < rows; ++y) {
					pd[x] += *ps;
					ps += stride;
				}
			}
			break;
		case ReduceMode::Avg:
			for (int x = 0; x < stride; ++x) {
				const uint8_t *ps = data + x;
				for (int y = 0; y < rows; ++y) {
					pd[x] += *ps;
					ps += stride;
				}
				pd[x] /= rows;
			}
			break;
		case ReduceMode::Max:
			for (int x = 0; x < stride; ++x) {
				const uint8_t *ps = data + x;
				pd[x] = *ps;
				ps += stride;
				for (int y = 1; y < rows; ++y) {
					if (*ps > pd[x])
						pd[x] = *ps;
					ps += stride;
				}
			}
			break;
		case ReduceMode::Min:
			for (int x = 0; x < stride; ++x) {
				const uint8_t *ps = data + x;
				pd[x] = *ps;
				ps += stride;
				for (int y = 1; y < rows; ++y) {
					if (*ps < pd[x])
						pd[x] = *ps;
					ps += stride;
				}
			}
			break;
		default:
			Critical("Mat::Reduce: unknown reduce mode.");
			return;
		}
		return;
	}

	outM.Reshape(rows, 1);
	float *pd = outM.data;
	switch (mode) {
	case ReduceMode::Sum:
		for (int y = 0; y < rows; ++y) {
			const uint8_t *ps = data + y * stride;
			for (int x = 0; x < stride; ++x) {
				pd[y] += *(ps++);
			}
		}
		break;
	case ReduceMode::Avg:
		for (int y = 0; y < rows; ++y) {
			const uint8_t *ps = data + y * stride;
			for (int x = 0; x < stride; ++x) {
				pd[y] += *(ps++);
			}
			pd[y] /= stride;
		}
		break;
	case ReduceMode::Max:
		for (int y = 0; y < rows; ++y) {
			const uint8_t *ps = data + y * stride;
			pd[y] = *(ps++);
			for (int x = 1; x < stride; ++x) {
				if (*ps > pd[y])
					pd[y] = *ps;
				++ps;
			}
		}
		break;
	case ReduceMode::Min:
		for (int y = 0; y < rows; ++y) {
			const uint8_t *ps = data + y * stride;
			pd[y] = *(ps++);
			for (int x = 1; x < stride; ++x) {
				if (*ps < pd[y])
					pd[y] = *ps;
				++ps;
			}
		}
		break;
	default:
		Critical("Mat::Reduce: unknown reduce mode.");
		return;
	}
}

void Mat::Repeat(Mat &outM, int ny, int nx) const {
	outM.Reshape(rows * ny, cols * nx, channels, false);
	int strideSrc = cols * channels;
	int strideDest = strideSrc * nx;

	for (int y = 0; y < rows; ++y) {
		const uint8_t *ps = data + strideSrc * y;
		uint8_t *pd = outM.data + strideDest * y;
		for (int k = 0; k < nx; ++k) {
			memcpy(pd, ps, strideSrc);
			pd += strideSrc;
		}
	}
	const uint8_t *ps = outM.data;
	int len = rows * strideDest;
	for (int y = 0; y < ny - 1; ++y) {
		uint8_t *pd = outM.data + (y + 1) * len;
		memcpy(pd, ps, len);
	}
}

//
// MatF Implementation
//

MatF::MatF(int rowNum, int colNum, MemoryArena *storage)
	: rows(rowNum), cols(colNum), arena(storage), data(nullptr) {
	size = rows * cols;
	if (arena) data = arena->Alloc<float>(size, false);
	else data = AllocAligned<float>(size);
	memset(data, 0, size * sizeof(float));
}

MatF::MatF(const float *sourceData, int rowNum, int colNum, MemoryArena *storage)
	: rows(rowNum), cols(colNum), arena(storage), data(nullptr) {
	size = rows * cols;
	if (arena) data = arena->Alloc<float>(size, false);
	else data = AllocAligned<float>(size);
	memcpy(data, sourceData, size * sizeof(float));
}

MatF::MatF(const MatF &m)
	: rows(m.rows), cols(m.cols), size(m.size), arena(m.arena) {
	if (arena) data = arena->Alloc<float>(size, false);
	else data = AllocAligned<float>(size);
	memcpy(data, m.data, size * sizeof(float));
}

MatF::MatF(MatF &&m) noexcept
	: rows(m.rows), cols(m.cols), size(m.size), arena(m.arena), data(m.data) {
	m.data = nullptr;
	m.arena = nullptr;
	m.rows = m.cols = m.size = 0;
}

MatF &MatF::operator=(const MatF &m) {
	rows = m.rows;
	cols = m.cols;
	size = m.size;
	if (data && !arena) FreeAligned(data);
	if (arena) data = arena->Alloc<float>(size, false);
	else data = AllocAligned<float>(size);
	memcpy(data, m.data, size * sizeof(float));
	return *this;
}

MatF &MatF::operator=(MatF &&m) noexcept {
	rows = m.rows;
	cols = m.cols;
	size = m.size;
	if (data && !arena) FreeAligned(data);
	if (arena == m.arena) {
		data = m.data;
		m.data = nullptr;
		m.arena = nullptr;
		m.rows = m.cols = m.size = 0;
	} else {
		if (arena) data = arena->Alloc<float>(size, false);
		else data = AllocAligned<float>(size);
		memcpy(data, m.data, size * sizeof(float));
	}
	return *this;
}

MatF &MatF::Reshape(int rowNum, int colNum, bool zero) {
	if (rowNum == rows && colNum == cols) {
		if (zero)
			memset(data, 0, size * sizeof(float));
		return *this;
	}

	rows = rowNum;
	cols = colNum;
	size = rows * cols;
	if (data && !arena) FreeAligned(data);
	if (arena) data = arena->Alloc<float>(size, false);
	else data = AllocAligned<float>(size);

	if (zero)
		memset(data, 0, size * sizeof(float));
	return *this;
}

MatF::~MatF() {
	if (data && !arena) FreeAligned(data);
}

void MatF::ManageExternalData(float *sourceData, int rowNum, int colNum) {
    if (data && !arena) FreeAligned(data);
    data = sourceData;
    rows = rowNum;
    cols = colNum;
    size = rows * cols;
    arena = &GImgLoadArena;
}

float MatF::Mean() const {
	float sum = 0.0f;
	const float *ptemp = data;
	for (int i = 0; i < size; ++i)
		sum += *(ptemp++);
	return sum / size;
}

void MatF::ReValue(float offset) {
	float *ptemp = data;
	for (int i = 0; i < size; ++i)
		*(ptemp++) += offset;
}

void MatF::ReValue(float scale, float offset) {
	float *ptemp = data;
	for (int i = 0; i < size; ++i) {
		*ptemp = *ptemp * scale + offset;
		++ptemp;
	}
}

void MatF::Lerp(const MatF& mat, float ratio) {
	if (mat.rows != rows || mat.cols != cols) {
		Critical("MatF::Lerp: cannot lerp between 2 different size matrix.");
		return;
	}
	const float *pm = mat.data;
	float *ps = data;
	for (int i = 0; i < size; ++i) {
		*ps = (1 - ratio) * (*ps) + ratio * (*(pm++));
		++ps;
	}
}

void MatF::CMul(const MatF &mat) {
	if (mat.rows != rows || mat.cols != cols) {
		Critical("MatF::CMul: cannot perform element wise multiplication between 2 different size matrix.");
		return;
	}
	const float *pm = mat.data;
	float *ps = data;
	for (int i = 0; i < size; ++i) {
		*(ps++) *= *(pm++);
	}
}

void MatF::CAdd(const MatF &mat, float scale, float offset) {
	if (mat.rows != rows || mat.cols != cols) {
		Critical("MatF::CAdd: cannot perform element wise addition between 2 different size matrix.");
		return;
	}
	const float *pm = mat.data;
	float *ps = data;
	for (int i = 0; i < size; ++i) {
		*(ps++) += (*(pm++) * scale + offset);
	}
}

void MatF::MinMaxLoc(float& min, float& max, Vector2i& minLoc, Vector2i& maxLoc) const {
	if (size <= 0)
		Critical("MatF::MinMaxLoc: cannot evalute target value on empty mat.");

	min = max = data[0];
	int minIdx = 0, maxIdx = 0;
	const float *ptemp = data + 1;
	for (int i = 1; i < size; ++i) {
		if (*ptemp < min) {
			min = *ptemp;
			minIdx = i;
		} else if (*ptemp > max) {
			max = *ptemp;
			maxIdx = i;
		}
		++ptemp;
	}
	minLoc.y = minIdx / cols;
	minLoc.x = minIdx - minLoc.y * cols;
	maxLoc.y = maxIdx / cols;
	maxLoc.x = maxIdx - maxLoc.y * cols;
}

void MatF::MinLoc(float &min, Vector2i &minLoc) const {
	if (size <= 0)
		Critical("MatF::MinLoc: cannot evalute target value on empty mat.");

	min = data[0];
	int minIdx = 0;
	const float *ptemp = data + 1;
	for (int i = 1; i < size; ++i) {
		if (*ptemp < min) {
			min = *ptemp;
			minIdx = i;
		}
		++ptemp;
	}
	minLoc.y = minIdx / cols;
	minLoc.x = minIdx - minLoc.y * cols;
}

void MatF::MaxLoc(float &max, Vector2i &maxLoc) const {
	if (size <= 0)
		Critical("MatF::MaxLoc: cannot evalute target value on empty mat.");

	max = data[0];
	int maxIdx = 0;
	const float *ptemp = data + 1;
	for (int i = 1; i < size; ++i) {
		if (*ptemp > max) {
			max = *ptemp;
			maxIdx = i;
		}
		++ptemp;
	}
	maxLoc.y = maxIdx / cols;
	maxLoc.x = maxIdx - maxLoc.y * cols;
}

void MatF::Split(std::vector<MatF>& mats, int channels, MemoryArena *targetArena) const {
	if (cols % channels != 0)
		Critical("MatF::Split: mat's column number is not multiple of channel number.");
	
	mats.clear();
	mats.resize(channels, MatF(targetArena));
	int tcols = cols / channels;
	for (int i = 0; i < channels; ++i)
		mats[i].Reshape(rows, tcols, false);
	const float *pt = data;
	// Allocate in stack, automatically freed.
	float **pts = ALLOCA(float*, channels);
	for (int k = 0; k < channels; ++k)
		pts[k] = mats[k].data;
	int tarlen = size / channels;
	for (int index = 0; index < tarlen; ++index) {
		for (int k = 0; k < channels; ++k)
			*(pts[k]++) = *(pt++);
	}
}

void MatF::Merge(const std::vector<MatF> &mats) {
	int channels = (int)mats.size();
	this->Reshape(mats[0].rows, mats[0].cols * channels, false);
	float *pt = data;
	// Allocate in stack, automatically freed.
	float **pts = ALLOCA(float*, channels);
	for (int k = 0; k < channels; ++k)
		pts[k] = mats[k].data;

	int tarlen = size / channels;
	for (int index = 0; index < tarlen; ++index) {
		for (int k = 0; k < channels; ++k)
			*(pt++) = *(pts[k]++);
	}
}

void MatF::Resize(MatF &dest, float sr, float sc, ResizeMode mode) const {
	int trows = (int)(rows * sr);
	int tcols = (int)(cols * sc);
	dest.Reshape(trows, tcols, false);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC1, data);
	cv::Mat cvDstM(trows, tcols, CV_32FC1, dest.data);
	int interp = cv::INTER_LINEAR;
	switch (mode) {
		case ResizeMode::Bicubic:
			interp = cv::INTER_CUBIC;
			break;
		case ResizeMode::Nearest:
			interp = cv::INTER_NEAREST;
			break;
		default:
			interp = cv::INTER_LINEAR;
			break;
	}
	cv::resize(cvOrgM, cvDstM, cv::Size(tcols, trows), 0, 0, interp);
#else
	if(mode == ResizeMode::Nearest) ResizeNearest(data, cols, rows, cols, dest.data, dest.cols, dest.rows, dest.cols, 1);
	else ResizeBilinear(data, cols, rows, cols, dest.data, dest.cols, dest.rows, dest.cols, 1);
#endif
}

void MatF::Resize(MatF &dest, int tr, int tc, ResizeMode mode) const {
	dest.Reshape(tr, tc, false);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC1, data);
	cv::Mat cvDstM(tr, tc, CV_32FC1, dest.data);
	int interp = cv::INTER_LINEAR;
	switch (mode) {
		case ResizeMode::Bicubic:
			interp = cv::INTER_CUBIC;
			break;
		case ResizeMode::Nearest:
			interp = cv::INTER_NEAREST;
			break;
		default:
			interp = cv::INTER_LINEAR;
			break;
	}
	cv::resize(cvOrgM, cvDstM, cv::Size(tc, tr), 0, 0, interp);
#else
	if(mode == ResizeMode::Nearest) ResizeNearest(data, cols, rows, cols, dest.data, dest.cols, dest.rows, dest.cols, 1);
	else ResizeBilinear(data, cols, rows, cols, dest.data, dest.cols, dest.rows, dest.cols, 1);
#endif
}

void MatF::Resize(MatF &dest, float sr, float sc, int channels, ResizeMode mode) const {
	if (cols % channels != 0)
		Critical("MatF::Resize: cannot resize for target channel number.");
	int ocols = cols / channels;
	int trows = (int)(rows * sr);
	int tcols = (int)(ocols * sc);
	dest.Reshape(trows, tcols * channels, false);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, ocols, CV_32FC(channels), data);
	cv::Mat cvDstM(trows, tcols, CV_32FC(channels), dest.data);
	int interp = cv::INTER_LINEAR;
	switch (mode) {
		case ResizeMode::Bicubic:
			interp = cv::INTER_CUBIC;
			break;
		case ResizeMode::Nearest:
			interp = cv::INTER_NEAREST;
			break;
		default:
			interp = cv::INTER_LINEAR;
			break;
	}
	cv::resize(cvOrgM, cvDstM, cv::Size(tcols, trows), 0, 0, interp);
#else
	if(mode == ResizeMode::Nearest) ResizeNearest(data, ocols, rows, cols, dest.data, tcols, dest.rows, dest.cols, channels);
	else ResizeBilinear(data, ocols, rows, cols, dest.data, tcols, dest.rows, dest.cols, channels);
#endif
}

void MatF::Resize(MatF &dest, int tr, int tc, int channels, ResizeMode mode) const {
	if (cols % channels != 0)
		Critical("MatF::Resize: cannot resize for target channel number.");
	dest.Reshape(tr, tc * channels, false);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols / channels, CV_32FC(channels), data);
	cv::Mat cvDstM(tr, tc, CV_32FC(channels), dest.data);
	int interp = cv::INTER_LINEAR;
	switch (mode) {
		case ResizeMode::Bicubic:
			interp = cv::INTER_CUBIC;
			break;
		case ResizeMode::Nearest:
			interp = cv::INTER_NEAREST;
			break;
		default:
			interp = cv::INTER_LINEAR;
			break;
	}
	cv::resize(cvOrgM, cvDstM, cv::Size(tc, tr), 0, 0, interp);
#else
	if(mode == ResizeMode::Nearest) ResizeNearest(data, cols / channels, rows, cols, dest.data, tc, dest.rows, dest.cols, channels);
	else ResizeBilinear(data, cols / channels, rows, cols, dest.data, tc, dest.rows, dest.cols, channels);
#endif
}

void MatF::ToMat(Mat& dest, int channels, float scale, float offset) const {
	if (cols % channels != 0)
		Warning("MatF::ToMat: channels number may be wrong, cannot be divided by cols.");

	dest.Reshape(rows, cols / channels, channels, false);
	uint8_t *pd = dest.data;
	const float *ps = data;
	for (int i = 0; i < size; ++i) {
		*(pd++) = (uint8_t)Clamp((int)(*(ps++) * scale + offset), 0, 255);
	}
}

void MatF::ToMatCF(MatCF& dest) const {
	dest.Reshape(rows, cols);
	const float *ps = data;
	float *pd = (float*)dest.data;
	for (int i = 0; i < size; ++i) {
		*pd = *(ps++);
		pd += 2;
	}
}

void MatF::Roi(const Bounds2i &bound, MatF &outM) const {
	int trows = bound.pMax.y - bound.pMin.y;
	int tcols = bound.pMax.x - bound.pMin.x;
	outM.Reshape(trows, tcols, false);
	for (int y = 0; y < trows; ++y) {
		const float *ps = data + (bound.pMin.y + y) * cols + bound.pMin.x;
		float *pd = outM.data + y * tcols;
		memcpy(pd, ps, tcols * sizeof(float));
	}
}

bool MatF::MakeBorder(MatF &outM, int left, int top, int right, int bottom, bool replicate) const {
	if (left == 0 && top == 0 && right == 0 && bottom == 0)
		return false;

	int trows = rows + top + bottom;
	int tcols = cols + left + right;
	outM.Reshape(trows, tcols);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC1, data);
	cv::Mat cvDstM(trows, tcols, CV_32FC1, outM.data);
	cv::copyMakeBorder(cvOrgM, cvDstM, top, bottom, left, right, replicate ? cv::BORDER_REPLICATE : cv::BORDER_CONSTANT);
#else
	if (replicate) {
		for (int y = 0; y < trows; ++y) {
			float *pd = outM.data + y * tcols + left;
			const float *ps = nullptr;
			if (y < top)
				ps = data;
			else if (y > top + rows - 1)
				ps = data + (rows - 1) * cols;
			else
				ps = data + (y - top) * cols;

			memcpy(pd, ps, cols * sizeof(float));
			float *ls = outM.data + y * tcols;
			float *rs = outM.data + y * tcols + tcols - right;
			for (int i = 0; i < left; ++i)
				*(ls++) = *ps;
			ps += (cols - 1);
			for (int i = 0; i < right; ++i)
				*(rs++) = *ps;
		}
	} else {
		for (int y = 0; y < rows; ++y) {
			const float *ps = data + y * cols;
			float *pd = outM.data + (y + top) * tcols + left;
			memcpy(pd, ps, cols * sizeof(float));
		}
	}
#endif
	return true;
}

void MatF::Reduce(MatF& outM, ReduceMode mode, bool toRow) const {
	if (size == 0) {
		Critical("MatF::Reduce: cannot reduce an empty mat.");
		return;
	}
	int stride = cols;

	if (toRow) {
		outM.Reshape(1, stride);
		float *pd = outM.data;
		switch (mode) {
		case ReduceMode::Sum:
			for (int x = 0; x < stride; ++x) {
				const float *ps = data + x;
				for (int y = 0; y < rows; ++y) {
					pd[x] += *ps;
					ps += stride;
				}
			}
			break;
		case ReduceMode::Avg:
			for (int x = 0; x < stride; ++x) {
				const float *ps = data + x;
				for (int y = 0; y < rows; ++y) {
					pd[x] += *ps;
					ps += stride;
				}
				pd[x] /= rows;
			}
			break;
		case ReduceMode::Max:
			for (int x = 0; x < stride; ++x) {
				const float *ps = data + x;
				pd[x] = *ps;
				ps += stride;
				for (int y = 1; y < rows; ++y) {
					if (*ps > pd[x])
						pd[x] = *ps;
					ps += stride;
				}
			}
			break;
		case ReduceMode::Min:
			for (int x = 0; x < stride; ++x) {
				const float *ps = data + x;
				pd[x] = *ps;
				ps += stride;
				for (int y = 1; y < rows; ++y) {
					if (*ps < pd[x])
						pd[x] = *ps;
					ps += stride;
				}
			}
			break;
		default:
			Critical("MatF::Reduce: unknown reduce mode.");
			return;
		}
		return;
	}

	outM.Reshape(rows, 1);
	float *pd = outM.data;
	switch (mode) {
	case ReduceMode::Sum:
		for (int y = 0; y < rows; ++y) {
			const float *ps = data + y * stride;
			for (int x = 0; x < stride; ++x) {
				pd[y] += *(ps++);
			}
		}
		break;
	case ReduceMode::Avg:
		for (int y = 0; y < rows; ++y) {
			const float *ps = data + y * stride;
			for (int x = 0; x < stride; ++x) {
				pd[y] += *(ps++);
			}
			pd[y] /= stride;
		}
		break;
	case ReduceMode::Max:
		for (int y = 0; y < rows; ++y) {
			const float *ps = data + y * stride;
			pd[y] = *(ps++);
			for (int x = 1; x < stride; ++x) {
				if (*ps > pd[y])
					pd[y] = *ps;
				++ps;
			}
		}
		break;
	case ReduceMode::Min:
		for (int y = 0; y < rows; ++y) {
			const float *ps = data + y * stride;
			pd[y] = *(ps++);
			for (int x = 1; x < stride; ++x) {
				if (*ps < pd[y])
					pd[y] = *ps;
				++ps;
			}
		}
		break;
	default:
		Critical("MatF::Reduce: unknown reduce mode.");
		return;
	}
}

void MatF::Repeat(MatF &outM, int ny, int nx) const {
	outM.Reshape(rows * ny, cols * nx, false);
	int strideSrc = cols;
	int strideDest = strideSrc * nx;

	for (int y = 0; y < rows; ++y) {
		const float *ps = data + strideSrc * y;
		float *pd = outM.data + strideDest * y;
		for (int k = 0; k < nx; ++k) {
			memcpy(pd, ps, strideSrc * sizeof(float));
			pd += strideSrc;
		}
	}
	const float *ps = outM.data;
	int len = rows * strideDest;
	for (int y = 0; y < ny - 1; ++y) {
		float *pd = outM.data + (y + 1) * len;
		memcpy(pd, ps, len * sizeof(float));
	}
}

void CwiseMul(const MatF &A, const MatF &B, MatF &C, float scale, float offset) {
	int rows = A.Rows();
	int cols = A.Cols();
	if (rows != B.Rows() || cols != B.Cols()) {
		Critical("CwiseMul: cannot perform element wise multiplication between 2 different size matrix.");
		return;
	}
	C.Reshape(rows, cols, false);

	const float *pa = A.Data();
	const float *pb = B.Data();
	float *pc = C.Data();
	for (int i = 0; i < A.Size(); ++i)
		*(pc++) = (*(pa++)) * (*(pb++)) * scale + offset;
}

void CWiseAdd(const MatF &A, const MatF &B, MatF &C, float scale, float offset) {
	int rows = A.Rows();
	int cols = A.Cols();
	if (rows != B.Rows() || cols != B.Cols()) {
		Critical("CWiseAdd: cannot perform element wise addition between 2 different size matrix.");
		return;
	}
	C.Reshape(rows, cols, false);

	const float *pa = A.Data();
	const float *pb = B.Data();
	float *pc = C.Data();
	for (int i = 0; i < A.Size(); ++i)
		*(pc++) = (*(pa++) + *(pb++)) * scale + offset;
}

void CWiseAddRecip(const MatF &A, const MatF &B, MatF &C, float scale, float offset) {
	int rows = A.Rows();
	int cols = A.Cols();
	if (rows != B.Rows() || cols != B.Cols()) {
		Critical("CWiseAddRecip: cannot perform element wise addition between 2 different size matrix.");
		return;
	}
	C.Reshape(rows, cols, false);

	const float *pa = A.Data();
	const float *pb = B.Data();
	float *pc = C.Data();
	for (int i = 0; i < A.Size(); ++i)
		*(pc++) = 1.0f / ((*(pa++) + *(pb++)) * scale + offset);
}

void CWiseLog(const MatF &A, MatF &B) {
	int rows = A.Rows();
	int cols = A.Cols();
	B.Reshape(rows, cols, false);
	const float *pa = A.Data();
	float *pb = B.Data();
	float temp;
	for (int i = 0; i < A.Size(); ++i) {
		temp = *(pa++);
		*(pb++) = (temp <= 0.0f ? -800.0f : std::log(temp));
	}
}

//
// MatCF Implementation
//

MatCF::MatCF(int rowNum, int colNum, MemoryArena *storage)
	: rows(rowNum), cols(colNum), arena(storage), data(nullptr) {
	size = rows * cols;
	if (arena) data = (ComplexF*)arena->Alloc<TempInit>(size, false);
	else data = (ComplexF*)AllocAligned<TempInit>(size);
	memset(data, 0, size * sizeof(ComplexF));
}

MatCF::MatCF(const ComplexF *sourceData, int rowNum, int colNum, MemoryArena *storage)
	: rows(rowNum), cols(colNum), arena(storage), data(nullptr) {
	size = rows * cols;
	if (arena) data = (ComplexF*)arena->Alloc<TempInit>(size, false);
	else data = (ComplexF*)AllocAligned<TempInit>(size);
	memcpy(data, sourceData, size * sizeof(ComplexF));
}

MatCF::MatCF(const MatCF &m)
	: rows(m.rows), cols(m.cols), size(m.size), arena(m.arena) {
	if (arena) data = (ComplexF*)arena->Alloc<TempInit>(size, false);
	else data = (ComplexF*)AllocAligned<TempInit>(size);
	memcpy(data, m.data, size * sizeof(ComplexF));
}

MatCF::MatCF(MatCF &&m) noexcept
	: rows(m.rows), cols(m.cols), size(m.size), arena(m.arena), data(m.data) {
	m.data = nullptr;
	m.arena = nullptr;
	m.rows = m.cols = m.size = 0;
}

MatCF &MatCF::operator=(const MatCF &m) {
	rows = m.rows;
	cols = m.cols;
	size = m.size;
	if (data && !arena) FreeAligned(data);
	if (arena) data = (ComplexF*)arena->Alloc<TempInit>(size, false);
	else data = (ComplexF*)AllocAligned<TempInit>(size);
	memcpy(data, m.data, size * sizeof(ComplexF));
	return *this;
}

MatCF &MatCF::operator=(MatCF &&m) noexcept {
	rows = m.rows;
	cols = m.cols;
	size = m.size;
	if (data && !arena) FreeAligned(data);
	if (arena == m.arena) {
		data = m.data;
		m.data = nullptr;
		m.arena = nullptr;
		m.rows = m.cols = m.size = 0;
	} else {
		if (arena) data = (ComplexF*)arena->Alloc<TempInit>(size, false);
		else data = (ComplexF*)AllocAligned<TempInit>(size);
		memcpy(data, m.data, size * sizeof(ComplexF));
	}
	return *this;
}

MatCF &MatCF::Reshape(int rowNum, int colNum, bool zero) {
	if (rowNum == rows && colNum == cols) {
		if (zero)
			memset(data, 0, size * sizeof(ComplexF));
		return *this;
	}

	rows = rowNum;
	cols = colNum;
	size = rows * cols;
	if (data && !arena) FreeAligned(data);
	if (arena) data = (ComplexF*)arena->Alloc<TempInit>(size, false);
	else data = (ComplexF*)AllocAligned<TempInit>(size);

	if (zero)
		memset(data, 0, size * sizeof(ComplexF));
	return *this;
}

MatCF::~MatCF() {
	if (data && !arena) FreeAligned(data);
}

void MatCF::ManageExternalData(CSRT::complex<float> *sourceData, int rowNum, int colNum) {
    if (data && !arena) FreeAligned(data);
    data = sourceData;
    rows = rowNum;
    cols = colNum;
    size = rows * cols;
    arena = &GImgLoadArena;
}

ComplexF MatCF::Mean() const {
	ComplexF sum(0.0f, 0.0f);
	ComplexF *ptemp = data;
	for (int i = 0; i < size; ++i)
		sum += *(ptemp++);
	return sum / (float)size;
}

void MatCF::OffsetValue(float offset) {
	float *ps = (float*)data;
	for (int i = 0; i < size; ++i) {
		*ps += offset;
		ps += 2;
	}
}

void MatCF::ScaleValue(float scale) {
	float *ps = (float*)data;
	int tarlen = size * 2;
	for (int i = 0; i < tarlen; ++i)
		*ps *= scale;
}

void MatCF::Lerp(const MatCF& mat, float ratio) {
	if (mat.rows != rows || mat.cols != cols) {
		Critical("MatCF::Lerp: cannot lerp between 2 different size matrix.");
		return;
	}
	float *ps = (float*)data;
	const float *pm = (const float*)mat.data;
	int tarlen = size * 2;
	for (int i = 0; i < tarlen; ++i) {
		*ps = (1 - ratio)*(*ps) + ratio * (*(pm++));
		++ps;
	}
}

void MatCF::CMul(const MatCF &mat) {
	if (mat.rows != rows || mat.cols != cols) {
		Critical("MatCF::CMul: cannot perform element wise multiplication between 2 different size matrix.");
		return;
	}
	const ComplexF *pm = mat.data;
	ComplexF *ps = data;
	for (int i = 0; i < size; ++i) {
		*(ps++) *= *(pm++);
	}
}

void MatCF::CAdd(const MatCF &mat, float scale, ComplexF offset) {
	if (mat.rows != rows || mat.cols != cols) {
		Critical("MatCF::CAdd: cannot perform element wise addition between 2 different size matrix.");
		return;
	}
	const ComplexF *pm = mat.data;
	ComplexF *ps = data;
	for (int i = 0; i < size; ++i)
		*(ps++) += (*(pm++) * scale + offset);
}

void MatCF::ToMatF(MatF& dest, ComplexMode mode) const {
	if (mode == ComplexMode::All)
		dest.Reshape(rows, cols * 2, false);
	else
		dest.Reshape(rows, cols, false);
	const float *ps = (float*)data;
	float *pd = dest.data;

	switch (mode) {
	case ComplexMode::Real:
		for (int i = 0; i < size; ++i) {
			*(pd++) = *ps;
			ps += 2;
		}
		break;
	case ComplexMode::Img:
		++ps;
		for (int i = 0; i < size; ++i) {
			*(pd++) = *ps;
			ps += 2;
		}
		break;
	case ComplexMode::Mag:
		float re, im;
		for (int i = 0; i < size; ++i) {
			re = *(ps++);
			im = *(ps++);
			*(pd++) = sqrt(re*re + im * im);
		}
		break;
	case ComplexMode::All:
		memcpy(pd, ps, dest.size * sizeof(float));
		break;
	default:
		Critical("MatCF::ToMatF: unknown mode for convertion.");
		return;
	}
}

void MatCF::Split(MatF &real, MatF &img) const {
	real.Reshape(rows, cols, false);
	img.Reshape(rows, cols, false);
	const float *ps = (float*)data;
	float *pr = real.data;
	float *pi = img.data;
	for (int i = 0; i < size; ++i) {
		*(pr++) = *(ps++);
		*(pi++) = *(ps++);
	}
}

void MatCF::Merge(const MatF &real, const MatF &img) {
	this->Reshape(real.rows, real.cols, false);
	float *ps = (float*)data;
	const float *pr = real.data;
	const float *pi = img.data;
	for (int i = 0; i < size; ++i) {
		*(ps++) = *(pr++);
		*(ps++) = *(pi++);
	}
}

void MatCF::Roi(const Bounds2i &bound, MatCF &outM) const {
	int trows = bound.pMax.y - bound.pMin.y;
	int tcols = bound.pMax.x - bound.pMin.x;
	outM.Reshape(trows, tcols, false);
	for (int y = 0; y < trows; ++y) {
		const ComplexF *ps = data + (bound.pMin.y + y) * cols + bound.pMin.x;
		ComplexF *pd = outM.data + y * tcols;
		memcpy(pd, ps, tcols * sizeof(ComplexF));
	}
}

bool MatCF::MakeBorder(MatCF &outM, int left, int top, int right, int bottom, bool replicate) const {
	if (left == 0 && top == 0 && right == 0 && bottom == 0)
		return false;

	int trows = rows + top + bottom;
	int tcols = cols + left + right;
	outM.Reshape(trows, tcols);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC2, data);
	cv::Mat cvDstM(trows, tcols, CV_32FC2, outM.data);
	cv::copyMakeBorder(cvOrgM, cvDstM, top, bottom, left, right, replicate ? cv::BORDER_REPLICATE : cv::BORDER_CONSTANT);
#else
	if (replicate) {
		for (int y = 0; y < trows; ++y) {
			ComplexF *pd = outM.data + y * tcols + left;
			const ComplexF *ps = nullptr;
			if (y < top)
				ps = data;
			else if (y > top + rows - 1)
				ps = data + (rows - 1) * cols;
			else
				ps = data + (y - top) * cols;

			memcpy(pd, ps, cols * sizeof(ComplexF));
			ComplexF *ls = outM.data + y * tcols;
			ComplexF *rs = outM.data + y * tcols + tcols - right;
			for (int i = 0; i < left; ++i)
				*(ls++) = *ps;
			ps += (cols - 1);
			for (int i = 0; i < right; ++i)
				*(rs++) = *ps;
		}
	} else {
		for (int y = 0; y < rows; ++y) {
			const ComplexF *ps = data + y * cols;
			ComplexF *pd = outM.data + (y + top) * tcols + left;
			memcpy(pd, ps, cols * sizeof(ComplexF));
		}
	}
#endif
	return true;
}

void MatCF::Reduce(MatCF& outM, ReduceMode mode, bool toRow) const {
	if (size == 0) {
		Critical("MatCF::Reduce: cannot reduce an empty mat.");
		return;
	}
	int stride = cols;

	if (toRow) {
		outM.Reshape(1, stride);
		ComplexF *pd = outM.data;
		switch (mode) {
		case ReduceMode::Sum:
			for (int x = 0; x < stride; ++x) {
				const ComplexF *ps = data + x;
				for (int y = 0; y < rows; ++y) {
					pd[x] += *ps;
					ps += stride;
				}
			}
			break;
		case ReduceMode::Avg:
			for (int x = 0; x < stride; ++x) {
				const ComplexF *ps = data + x;
				for (int y = 0; y < rows; ++y) {
					pd[x] += *ps;
					ps += stride;
				}
				pd[x] /= rows;
			}
			break;
		case ReduceMode::Max:
			Warning("MatCF::Reduce: max reduce mode is not supported.");
			break;
		case ReduceMode::Min:
			Warning("MatCF::Reduce: min reduce mode is not supported.");
			break;
		default:
			Critical("MatCF::Reduce: unknown reduce mode.");
			return;
		}
		return;
	}

	outM.Reshape(rows, 1);
	ComplexF *pd = outM.data;
	switch (mode) {
	case ReduceMode::Sum:
		for (int y = 0; y < rows; ++y) {
			const ComplexF *ps = data + y * stride;
			for (int x = 0; x < stride; ++x) {
				pd[y] += *(ps++);
			}
		}
		break;
	case ReduceMode::Avg:
		for (int y = 0; y < rows; ++y) {
			const ComplexF *ps = data + y * stride;
			for (int x = 0; x < stride; ++x) {
				pd[y] += *(ps++);
			}
			pd[y] /= stride;
		}
		break;
	case ReduceMode::Max:
		Warning("MatCF::Reduce: max reduce mode is not supported.");
		break;
	case ReduceMode::Min:
		Warning("MatCF::Reduce: min reduce mode is not supported.");
		break;
	default:
		Critical("MatCF::Reduce: unknown reduce mode.");
		return;
	}
}

void MatCF::Repeat(MatCF &outM, int ny, int nx) const {
	outM.Reshape(rows * ny, cols * nx, false);
	int strideSrc = cols;
	int strideDest = strideSrc * nx;

	for (int y = 0; y < rows; ++y) {
		const ComplexF *ps = data + strideSrc * y;
		ComplexF *pd = outM.data + strideDest * y;
		for (int k = 0; k < nx; ++k) {
			memcpy(pd, ps, strideSrc * sizeof(ComplexF));
			pd += strideSrc;
		}
	}
	const ComplexF *ps = outM.data;
	int len = rows * strideDest;
	for (int y = 0; y < ny - 1; ++y) {
		ComplexF *pd = outM.data + (y + 1) * len;
		memcpy(pd, ps, len * sizeof(ComplexF));
	}
}

//
// Save mat data to loca jpeg image file.
//

void SaveToFile(const std::string &fileName, const Mat &mat) {
	if (mat.Size() == 0) {
		Error("SaveToFile: cannot save empty mat data.");
		return;
	}
	int res = stbi_write_jpg((fileName + ".jpg").c_str(), mat.Cols(), mat.Rows(), mat.Channels(), mat.Data(), 80);
	if (res == 0)
		Error("SaveToFile: cannot save mat data to path \"" + fileName + "\".");
}

void SaveToFile(const std::string &fileName, const MatF &mat) {
	if (mat.Size() == 0) {
		Error("SaveToFile: cannot save empty mat data.");
		return;
	}

	// Normalize.
	float maxValue, minValue;
	Vector2i maxLoc, minLoc;
	mat.MinMaxLoc(minValue, maxValue, minLoc, maxLoc);
	float scale = 1.0f / (maxValue - minValue);
	MatF norm;
	norm = mat;
	float *pd = norm.Data();
	for (int i = 0; i < norm.Size(); ++i) {
		*pd = (*pd - minValue) * scale;
		++pd;
	}
	Mat img;
	norm.ToMat(img, 1, 255.0f, 0.0f);
	SaveToFile(fileName, img);
}

void SaveToFile(const std::string &fileName, const MatCF &mat, ComplexMode mode) {
	if (mat.Size() == 0) {
		Error("SaveToFile: cannot save empty mat data.");
		return;
	}

	MatF img;
	mat.ToMatF(img, mode);
	SaveToFile(fileName, img);
}

//
// Print mat data to logger.
//

void PrintMat(const Mat &mat) {
	if (mat.Size() == 0) return;
	Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> egMat(mat.Data(), mat.Rows(), mat.Cols());
	Eigen::IOFormat format(3, 0, ", ", "\n", "", "", "[", "]");
	std::stringstream ss;
	ss << egMat.format(format) << "\n";
	Info("Matrix data: \n" + ss.str());
}

void PrintMat(const MatF &mat) {
	if (mat.Size() == 0) return;
	Eigen::Map<Eigen::MatrixXf> egMat(mat.Data(), mat.Rows(), mat.Cols());
	Eigen::IOFormat format(3, 0, ", ", "\n", "", "", "[", "]");
	std::stringstream ss;
	ss << egMat.format(format) << "\n";
	Info("Matrix data: \n" + ss.str());
}

void PrintMat(const MatCF &mat) {
	if (mat.Size() == 0) return;
	Eigen::Map<Eigen::MatrixXcf> egMat((std::complex<float>*)mat.Data(), mat.Rows(), mat.Cols());
	Eigen::IOFormat format(3, 0, ", ", "\n", "", "", "[", "]");
	std::stringstream ss;
	ss << egMat.format(format) << "\n";
	Info("Matrix data: \n" + ss.str());
}

}	// namespace CSRT
