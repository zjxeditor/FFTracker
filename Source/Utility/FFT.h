//
// Provide fast fft and inv fft operations with the help of fftw.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_FFT_H
#define CSRT_FFT_H

#include "../CSRT.h"
#include "Mat.h"
#include "fftw3.h"

namespace CSRT {

// Thread safe fft helper.
class FFT {
public:
	FFT();
	~FFT();
	void Initialize();

	// FFT Public Methods

	// 1d fft and inv fft.
	// Note, for long 1d array, please use 'FFT2...' methods for fast computation.
	void FFT1(const float *in, ComplexF *out, int len);
	void FFT1(const ComplexF *in, ComplexF *out, int len);
	void FFTInv1(const ComplexF *in, float *out, int len, bool scale = true);
	void FFTInv1(const ComplexF *in, ComplexF *out, int len, bool scale = true);
	// 2d fft and inv fft.
	void FFT2(const MatF &inM, MatCF &outM);
	void FFT2(const MatCF &inM, MatCF &outM);
	void FFTInv2(const MatCF &inM, MatF &outM, bool scale = true);
	void FFTInv2(const MatCF &inM, MatCF &outM, bool scale = true);
	// 1d fft and inv fft for each row.
	void FFT2Row(const MatF &inM, MatCF &outM);
	void FFT2Row(const MatCF &inM, MatCF &outM);
	void FFTInv2Row(const MatCF &inM, MatF &outM, bool scale = true);
	void FFTInv2Row(const MatCF &inM, MatCF &outM, bool scale = true);
	// 2d matrix fast transose.
	void Transpose(const MatF &inM, MatF &outM);
	void Transpose(float *in, float *out, int rows, int cols);

private:
	inline fftwf_complex * fftw_cast(const ComplexF *p) {
		return const_cast<fftwf_complex*>(reinterpret_cast<const fftwf_complex*>(p));
	}

	// Temporary structure for fast complex initialization.
	struct TempInit { float a, b; };	
	// Cache all created plans for reuse.
	std::unordered_map<std::string, fftwf_plan> planCache;
	// Planning of fftw is not thread safe.
	std::mutex FFTWMutex;

	bool initialized;
	static bool SingleInstance;
};

// Global FFT helper.
extern FFT GFFT;

}	// namespace CSRT

#endif	// CSRT_FFT_H

