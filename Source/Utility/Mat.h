//
// Baic image operations.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_MAT_H
#define CSRT_MAT_H

#include "../CSRT.h"
#include "Geometry.h"
#include "Memory.h"

namespace CSRT {

enum class ReduceMode {
	Sum = 0,
	Avg = 1,
	Max = 2,
	Min = 3
};

enum class BorderType {
	Zero = 0,
	Replicate = 1,
	Reflect = 2,
	Wrap = 3
};

enum class ComplexMode {
	Real = 0,
	Img = 1,
	Mag = 2,
	All = 3
};

enum class ResizeMode {
	Bilinear = 0,
	Bicubic = 1,
	Nearest = 2
};

inline int ExtrapolateBorder(int index, int len, BorderType type) {
	switch (type) {
	case BorderType::Replicate:
		return Clamp(index, 0, len - 1);
	case BorderType::Reflect:
		if (index < 0) return -index - 1;
		return 2 * len - index - 1;
	case BorderType::Wrap:
		if (index < 0) return index + len;
		return index - len;
	default:
		return -1;
	}
}

// Memory Arena Help Class
class ImageMemoryArena {
public:
	ImageMemoryArena();
	~ImageMemoryArena();
	void Initialize();
	void ResetImgLoadArena();
private:
	bool initialized;
	static bool SingleInstance;
};

// Global image arena management.
extern ImageMemoryArena GImageMemoryArena;

class MatF;

// Mat Class
class Mat {
	friend class MatF;

public:
	// Mat Public Methods
	Mat(MemoryArena *storage = nullptr) : rows(0), cols(0), channels(0), size(0), arena(storage), data(nullptr) {}
	Mat(int rowNum, int colNum, int channelNum, MemoryArena *storage = nullptr);
	Mat(const std::string &filePath);
	Mat(const uint8_t *sourceData, int rowNum, int colNum, int channelNum, MemoryArena *storage = nullptr);
	Mat(const Mat &m);
	Mat(Mat &&m) noexcept;
	Mat &operator=(const Mat &m);
	Mat &operator=(Mat &&m) noexcept;
	Mat &Reshape(int rowNum, int colNum, int channelNum, bool zero = true);
	~Mat();

	// Take control the external data. Just represent external data as Mat.
	void ManageExternalData(uint8_t *sourceData, int rowNum, int colNum, int channelNum);

	inline uint8_t Get(int row, int col, int channel) const {
		return *(data + row * cols * channels + col * channels + channel);
	}
	inline uint8_t &Get(int row, int col, int channel) {
		return *(data + row * cols * channels + col * channels + channel);
	}

	float Mean() const;
	void ReValue(float offset);
	void ReValue(float scale, float offset);
	void MinMaxLoc(uint8_t &min, uint8_t &max, Vector2i &minLoc, Vector2i &maxLoc) const;
	void MinLoc(uint8_t &min, Vector2i &minLoc) const;
	void MaxLoc(uint8_t &max, Vector2i &maxLoc) const;

	inline int Rows() const { return rows; }
	inline int Cols() const { return cols; }
	inline int Channels() const { return channels; }
	inline int Size() const { return size; }
	inline uint8_t *Data() const { return data; }

	void Split(std::vector<Mat> &mats, MemoryArena *targetArena = nullptr) const;
	void Merge(const std::vector<Mat> &mats);
	void Resize(Mat &dest, float sr, float sc, ResizeMode mode = ResizeMode::Bilinear) const;
	void Resize(Mat &dest, int tr, int tc, ResizeMode mode = ResizeMode::Bilinear) const;

	// Convert from rgb space to hsv space. All value are between [0, 255].
	void RGBToHSV(Mat &dest) const;
	// Convert from hsv space to rgb space. All value are between [0, 255].
	void HSVToRGB(Mat &dest) const;
	// Convert from rgb space to gray space. All value are between [0, 255].
	void RGBToGray(Mat &dest) const;
	// Convert from gray space to rgb space. All value are between [0, 255].
	void GrayToRGB(Mat &dest) const;
	// Convert from unsigned char type to float type.
	void ToMatF(MatF &dest, float scale = 1.0f, float offset = 0.0f) const;
	// Get the region of interest.
	void Roi(const Bounds2i &bound, Mat &outM) const;
	// Add borders. Border area is set to 0 by default. 
	// Otherwise, set the 'replicate' flag to 'true' to replicate border values.
	// Return 'true' for success operation.
	bool MakeBorder(Mat &outM, int left, int top, int right, int bottom, bool replicate = false) const;
	// Reduce matrix.
	void Reduce(MatF &outM, ReduceMode mode, bool toRow) const;
	// Repeat matrix.
	void Repeat(Mat &outM, int ny, int nx) const;

private:
	// Mat Private Data
	int rows, cols, channels, size;
	MemoryArena *arena;
	// Interleaved RGB. Currently not suport alpha channel.
	uint8_t* data;
};

class MatCF;
class FFT;

class MatF {
	friend class Mat;
	friend class MatCF;
	friend class FFT;

public:
	// MatF Public Methods
	MatF(MemoryArena *storage = nullptr) : rows(0), cols(0), size(0), arena(storage), data(nullptr) {}
	MatF(int rowNum, int colNum, MemoryArena *storage = nullptr);
	MatF(const float *sourceData, int rowNum, int colNum, MemoryArena *storage = nullptr);
	MatF(const MatF &m);
	MatF(MatF &&m) noexcept;
	MatF &operator=(const MatF &m);
	MatF &operator=(MatF &&m) noexcept;
	MatF &Reshape(int rowNum, int colNum, bool zero = true);
	~MatF();

    // Take control the external data. Just represent external data as MatF.
    void ManageExternalData(float *sourceData, int rowNum, int colNum);

	inline float Get(int row, int col) const {
		return *(data + row * cols + col);
	}
	inline float &Get(int row, int col) {
		return *(data + row * cols + col);
	}

	float Mean() const;
	void ReValue(float offset);
	void ReValue(float scale, float offset);
	void Lerp(const MatF &mat, float ratio);
	void CMul(const MatF &mat);
	void CAdd(const MatF &mat, float scale, float offset);

	void MinMaxLoc(float &min, float &max, Vector2i &minLoc, Vector2i &maxLoc) const;
	void MinLoc(float &min, Vector2i &minLoc) const;
	void MaxLoc(float &max, Vector2i &maxLoc) const;

	inline int Rows() const { return rows; }
	inline int Cols() const { return cols; }
	inline int Size() const { return size; }
	inline float *Data() const { return data; }

	void Split(std::vector<MatF> &mats, int channels, MemoryArena *targetArena = nullptr) const;
	void Merge(const std::vector<MatF> &mats);
	void Resize(MatF &dest, float sr, float sc, ResizeMode mode = ResizeMode::Bilinear) const;
	void Resize(MatF &dest, int tr, int tc, ResizeMode mode = ResizeMode::Bilinear) const;
	void Resize(MatF &dest, float sr, float sc, int channels, ResizeMode mode = ResizeMode::Bilinear) const;
	void Resize(MatF &dest, int tr, int tc, int channels, ResizeMode mode = ResizeMode::Bilinear) const;

	// Convert from float type to unsigned char type.
	void ToMat(Mat &dest, int channels = 1, float scale = 1.0f, float offset = 0.0f) const;
	// Convert from float type to complex float type.
	void ToMatCF(MatCF &dest) const;
	// Get the region of interest.
	void Roi(const Bounds2i &bound, MatF &outM) const;
	// Add borders. Border area is set to 0 by default. 
	// Otherwise, set the 'replicate' flag to 'true' to replicate border values.
	// Return 'true' for success operation.
	bool MakeBorder(MatF &outM, int left, int top, int right, int bottom, bool replicate = false) const;
	// Reduce matrix.
	void Reduce(MatF &outM, ReduceMode mode, bool toRow) const;
	// Repeat matrix.
	void Repeat(MatF &outM, int ny, int nx) const;

private:
	// MatF Private Data
	int rows, cols, size;
	MemoryArena *arena;
	float* data;
};

class MatCF {
	friend class MatF;

public:
	// MatF Public Methods
	MatCF(MemoryArena *storage = nullptr) : rows(0), cols(0), size(0), arena(storage), data(nullptr) {}
	MatCF(int rowNum, int colNum, MemoryArena *storage = nullptr);
	MatCF(const ComplexF *sourceData, int rowNum, int colNum, MemoryArena *storage = nullptr);
	MatCF(const MatCF &m);
	MatCF(MatCF &&m) noexcept;
	MatCF &operator=(const MatCF &m);
	MatCF &operator=(MatCF &&m) noexcept;
	MatCF &Reshape(int rowNum, int colNum, bool zero = true);
	~MatCF();

    // Take control the external data. Just represent external data as MatCF.
    void ManageExternalData(ComplexF *sourceData, int rowNum, int colNum);

	inline ComplexF Get(int row, int col) const {
		return *(data + row * cols + col);
	}
	inline ComplexF &Get(int row, int col) {
		return *(data + row * cols + col);
	}

	ComplexF Mean() const;
	void OffsetValue(float offset);
	void ScaleValue(float scale);
	void Lerp(const MatCF &mat, float ratio);
	void CMul(const MatCF &mat);
	void CAdd(const MatCF &mat, float scale, ComplexF offset);

	inline int Rows() const { return rows; }
	inline int Cols() const { return cols; }
	inline int Size() const { return size; }
	inline ComplexF *Data() const { return data; }

	// Split into real and img parts.
	void Split(MatF &real, MatF &img) const;
	// Merge from real and img parts.
	void Merge(const MatF &real, const MatF &img);

	// Convert from complex float type to float type.
	void ToMatF(MatF &dest, ComplexMode mode) const;
	// Get the region of interest.
	void Roi(const Bounds2i &bound, MatCF &outM) const;
	// Add borders. Border area is set to 0 by default. 
	// Otherwise, set the 'replicate' flag to 'true' to replicate border values.
	// Return 'true' for success operation.
	bool MakeBorder(MatCF &outM, int left, int top, int right, int bottom, bool replicate = false) const;
	// Reduce matrix.
	void Reduce(MatCF &outM, ReduceMode mode, bool toRow) const;
	// Repeat matrix.
	void Repeat(MatCF &outM, int ny, int nx) const;

private:
	// MatCF Private Data
	struct TempInit { float a, b; };	// Temporary structure for fast initialization.
	int rows, cols, size;
	MemoryArena *arena;
	ComplexF* data;
};

void CwiseMul(const MatF &A, const MatF &B, MatF &C, float scale = 1.0f, float offset = 0.0f);
void CWiseAdd(const MatF &A, const MatF &B, MatF &C, float scale = 1.0f, float offset = 0.0f);
void CWiseAddRecip(const MatF &A, const MatF &B, MatF &C, float scale = 1.0f, float offset = 0.0f);
void CWiseLog(const MatF &A, MatF &B);

//
// Debug methods. Performance inefficient. Do not use them in release mode.
//

// Save to mat data to local jpeg image. Note, fileName should not contain extension.
void SaveToFile(const std::string &fileName, const Mat &mat);
void SaveToFile(const std::string &fileName, const MatF &mat);
void SaveToFile(const std::string &fileName, const MatCF &mat, ComplexMode mode = ComplexMode::Mag);

// Print the mat data to logger.
void PrintMat(const Mat &mat);
void PrintMat(const MatF &mat);
void PrintMat(const MatCF &mat);

}	// namespace CSRT

#endif	// CSRT_MAT_H