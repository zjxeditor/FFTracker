//
// Basic header file for other components which defines global include files,
// maros, class references and utility functions.
//

#if defined(_MSC_VER)
#define NOMINMAX
#define _SCL_SECURE_NO_WARNINGS
#pragma once
#endif

#ifndef CSRT_CSRT_H
#define CSRT_CSRT_H

#ifdef _WIN32
#ifdef CSRT_EXPORT
#define CSRT_API __declspec(dllexport)
#else
#define CSRT_API __declspec(dllimport)
#endif
#else
#define CSRT_API
#endif

//
// Global Include Files
//

#include <algorithm>
#include <assert.h>
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdint.h>
#include <string.h>
#include <string>
#include <type_traits>
#include <vector>
#include <exception>
#include <chrono>
#include <complex>
#include <cstring>

#if defined(_MSC_VER)
#include <float.h>
#include <intrin.h>
#pragma warning(disable : 4305)  // double constant assigned to float
#pragma warning(disable : 4244)  // int -> float conversion
#pragma warning(disable : 4843)  // double -> float conversion
#endif

// Major Math Library
// Note, we use row major matrix operation for better image processing.
// Eigen library is used for complex matrix operations. Map functions will
// be used to intergrate the native data.
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "../ThirdParty/Eigen/Eigen/Dense"
// Logging support.
#include "../ThirdParty/Spdlog/include/spdlog/spdlog.h"

#define ALLOCA(TYPE, COUNT) (TYPE *) alloca((COUNT) * sizeof(TYPE))
#ifndef CSRT_L1_CACHE_LINE_SIZE
#define CSRT_L1_CACHE_LINE_SIZE 64
#endif

// Measure the best fft in the initial frame.
#define MEASURE_BEST_FFT

// Pre-defined hog feature count. Optimize the hog computation method. Choose 18, 27, 31 or 32. 
#define HOG_FEATURE_COUNT 31

namespace CSRT {

template<typename T>
class complex;

// Choose default complex operation implementation. Custom complex implementation is faster.
#define ComplexF complex<float>
#define Conj(a) a.conj()
//#define ComplexF std::complex<float>
//#define Conj(a) std::conj(a)

//
// Timing Function
//

typedef std::chrono::high_resolution_clock::time_point TimePt;
inline TimePt TimeNow() { return std::chrono::high_resolution_clock::now(); }
inline int64_t Duration(TimePt start, TimePt end) { return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); }

//
// Global Logging
//

extern CSRT_API std::shared_ptr<spdlog::logger> Logger;

// Create logger.
inline void CreateLogger(std::string filePath = "") {
	try {
		if (filePath.empty()) Logger = spdlog::stdout_color_mt("console");
		else Logger = spdlog::basic_logger_mt("file", filePath);
	} catch(const spdlog::spdlog_ex &ex) {
		throw std::runtime_error(ex.what());
	}
}

// Release and close all loggers.
inline void ClearLogger() {
	spdlog::drop_all();
	Logger.reset();
}

inline void Info(const std::string& message) {
	Logger->info(message);
}

inline void Warning(const std::string& message) {
	Logger->warn(message);
}

inline void Error(const std::string& message) {
	Logger->error(message);
}

inline void Critical(const std::string& message) {
	Logger->critical(message);
	throw std::runtime_error(message);
}

//
// Global Constants
//

static constexpr float MaxFloat = std::numeric_limits<float>::max();
static constexpr float Infinity = std::numeric_limits<float>::infinity();
static constexpr float Pi = 3.14159265358979323846;
static constexpr float InvPi = 0.31830988618379067154;
static constexpr float Inv2Pi = 0.15915494309189533577;
static constexpr float Inv4Pi = 0.07957747154594766788;
static constexpr float PiOver2 = 1.57079632679489661923;
static constexpr float PiOver4 = 0.78539816339744830961;
static constexpr float Sqrt2 = 1.41421356237309504880;

//
// Global Inline Functions
//

inline uint32_t FloatToBits(float f) {
	uint32_t ui;
	memcpy(&ui, &f, sizeof(float));
	return ui;
}

inline float BitsToFloat(uint32_t ui) {
	float f;
	memcpy(&f, &ui, sizeof(uint32_t));
	return f;
}

inline uint64_t FloatToBits(double f) {
	uint64_t ui;
	memcpy(&ui, &f, sizeof(double));
	return ui;
}

inline double BitsToFloat(uint64_t ui) {
	double f;
	memcpy(&f, &ui, sizeof(uint64_t));
	return f;
}

template <typename T, typename U, typename V>
inline T Clamp(T val, U low, V high) {
	if (val < low)
		return low;
	else if (val > high)
		return high;
	else
		return val;
}

template <typename T>
inline T Mod(T a, T b) {
	T result = a - (a / b) * b;
	return (T)((result < 0) ? result + b : result);
}

template <>
inline float Mod(float a, float b) {
	return std::fmod(a, b);
}

inline float Radians(float deg) { return (Pi / 180) * deg; }

inline float Degrees(float rad) { return (180 / Pi) * rad; }

inline float Log2(float x) {
	const float invLog2 = 1.442695040888963387004650940071;
	return std::log(x) * invLog2;
}

inline int Log2Int(uint32_t v) {
#if defined(_MSC_VER)
	unsigned long lz = 0;
	if (_BitScanReverse(&lz, v)) return lz;
	return 0;
#else
	return 31 - __builtin_clz(v);
#endif
}

inline int Log2Int(int32_t v) { return Log2Int((uint32_t)v); }

inline int Log2Int(uint64_t v) {
#if defined(_MSC_VER)
	unsigned long lz = 0;
#if defined(_WIN64)
	_BitScanReverse64(&lz, v);
#else
	if (_BitScanReverse(&lz, v >> 32))
		lz += 32;
	else
		_BitScanReverse(&lz, v & 0xffffffff);
#endif // _WIN64
	return lz;
#else  // PBRT_IS_MSVC
	return 63 - __builtin_clzll(v);
#endif
}

inline int Log2Int(int64_t v) { return Log2Int((uint64_t)v); }

template <typename T>
inline constexpr  bool IsPowerOf2(T v) {
	return v && !(v & (v - 1));
}

inline int32_t RoundUpPow2(int32_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

inline int64_t RoundUpPow2(int64_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	return v + 1;
}

inline float Lerp(float t, float v1, float v2) { return (1 - t) * v1 + t * v2; }

inline bool Quadratic(float a, float b, float c, float *t0, float *t1) {
	// Find quadratic discriminant
	double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
	if (discrim < 0) return false;
	double rootDiscrim = std::sqrt(discrim);

	// Compute quadratic _t_ values
	double q;
	if (b < 0)
		q = -.5 * (b - rootDiscrim);
	else
		q = -.5 * (b + rootDiscrim);
	*t0 = q / a;
	*t1 = c / q;
	if (*t0 > *t1) std::swap(*t0, *t1);
	return true;
}

inline float KernelEpan(float x) {
	return (x <= 1.0f) ? (2.0f / Pi)*(1 - x) : 0.0f;
}

} // namespace CSRT

#endif // CSRT_CSRT_H
