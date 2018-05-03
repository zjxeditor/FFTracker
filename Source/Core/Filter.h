//
// Filter related operations.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_FILTER_H
#define CSRT_FILTER_H

#include "../CSRT.h"
#include "../Utility/Geometry.h"
#include "../Utility/Mat.h"

namespace CSRT {

enum class SubpixelDirection {
	Horizontal = 0, 
	Vertical
};

class Filter {
public:
	Filter();
	~Filter();
	void Initialize();

	// Common filter kernel window generators.
	float ModifiedBessel(int order, float x);
	void ChebWin(MatF &window, int N, float atten);
	void HannWin(MatF &window, const Vector2i &size);
	void KaiserWin(MatF &window, const Vector2i &size, float alpha);
	void ChebyshevWin(MatF &window, const Vector2i &size, float attenuation);

	void CircShift(const MatF &src, MatF &dest, int dx, int dy);
	// Create 2D Gaussian peak, convert to Fourier space.
	void GaussianShapedLabels(MatCF &dest, float sigma, int w, int h);
	// FFT a collection of matrix.
	void FFT2Collection(const std::vector<MatF> &inMs, std::vector<MatCF> &outMs, MemoryArena *targetArena = nullptr);
	// Complex matrix element-wise devision.
	void DivideComplexMatrices(const MatCF &A, const MatCF &B, MatCF &dest, ComplexF boff = ComplexF(0.0f, 0.0f));
	// Complex matrix element-wise multiplication.
	// The function, together with FFT2() and FFTInv2(), may be used to calculate convolution (pass conjB=false ) 
	// or correlation (pass conjB=true ) of two arrays rapidly. 
	void MulSpectrums(const MatCF &A, const MatCF &B, MatCF &dest, bool conjB = false);
	// Get the specific sub window data. Clone.
	void GetSubWindow(const Mat &inM, Mat &outM, const Vector2i &center, int w, int h, Bounds2i *validBounds = nullptr);
	void GetSubWindow(const MatF &inM, MatF &outM, const Vector2i &center, int w, int h, Bounds2i *validBounds = nullptr);
	void GetSubWindow(const MatCF &inM, MatCF &outM, const Vector2i &center, int w, int h, Bounds2i *validBounds = nullptr);
	float SubpixelPeak(const MatF &inM, const Vector2i &p, SubpixelDirection direction);

	// Perform 2d direct correlation. If the kernel is sysmetry, correlation is equal to convolution.
	void Correlation2D(const MatF &inM, const MatF &kernelM, MatF &outM, BorderType borderType = BorderType::Replicate);

	// Perform 2d matrix dilate operation.
	void Dilate2D(const MatF &inM, const MatF &kernelM, MatF &outM, BorderType borderType = BorderType::Replicate);

	// Perform 2d matrix erode operation.
	void Erode2D(const MatF &inM, const MatF &kernelM, MatF &outM, BorderType borderType = BorderType::Replicate);

private:
	inline static float ChebPoly(int n, float x) {
		float res;
		if (fabs(x) <= 1)
			res = cos(n*acos(x));
		else
			res = cosh(n*acosh(x));
		return res;
	}

	inline static int ReflectIndex(int index, int len) {
		if (index < 0) return -index - 1;
		return 2 * len - index - 1;
	}

	inline static int WrapIndex(int index, int len) {
		if (index < 0) return index + len;
		return index - len;
	}

	bool initialized;
	static bool SingleInstance;
};

// Global Filter Helper.
extern Filter GFilter;

}	// namespace CSRT

#endif	// CSRT_FILTER_H