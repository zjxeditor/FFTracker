//
// Provide basic geometry support.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_GEOMETRY_H
#define CSRT_GEOMETRY_H

#include "../CSRT.h"
#include "StringPrint.h"
#include <iterator>

namespace CSRT {

template<typename T>
inline bool isNaN(const T x) {
	return std::isnan(x);
}

template<>
inline bool isNaN(const int x) {
	return false;
}


template<typename T>
class Vector3;

//
// Vector2 Declaration
//

template<typename T>
class Vector2 {
public:
	// Vector2 Public Methods
	Vector2() { x = y = 0; }
	Vector2(T xx, T yy) : x(xx), y(yy) {}
	bool HasNaNs() const { return isNaN(x) || isNaN(y); }
	explicit Vector2(const Vector3<T> &v);

	Vector2<T> operator+(const Vector2<T> &v) const {
		return Vector2(x + v.x, y + v.y);
	}
	Vector2<T> &operator+=(const Vector2<T> &v) {
		x += v.x;
		y += v.y;
		return *this;
	}
	Vector2<T> operator-(const Vector2<T> &v) const {
		return Vector2(x - v.x, y - v.y);
	}
	Vector2<T> &operator-=(const Vector2<T> &v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}
	Vector2<T> operator-() const { return Vector2<T>(-x, -y); }
	bool operator==(const Vector2<T> &v) const { return x == v.x && y == v.y; }
	bool operator!=(const Vector2<T> &v) const { return x != v.x || y != v.y; }

	template<typename U>
	Vector2<T> operator*(U f) const {
		return Vector2<T>(f * x, f * y);
	}
	template<typename U>
	Vector2<T> &operator*=(U f) {
		x *= f;
		y *= f;
		return *this;
	}
	template<typename U>
	Vector2<T> operator/(U f) const {
		float inv = 1.0f / f;
		return Vector2<T>(x * inv, y * inv);
	}
	template<typename U>
	Vector2<T> &operator/=(U f) {
		float inv = 1.0f / f;
		x *= inv;
		y *= inv;
		return *this;
	}

	T operator[](int i) const {
		if (i == 0) return x;
		return y;
	}
	T &operator[](int i) {
		if (i == 0) return x;
		return y;
	}

	float LengthSquared() const { return x * x + y * y; }
	float Length() const { return std::sqrt(LengthSquared()); }

	// Vector2 Public Data
	T x, y;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
	os << "[ " << v.x << ", " << v.y << " ]";
	return os;
}
template<>
inline std::ostream &operator<<(std::ostream &os, const Vector2<float> &v) {
	os << StringPrintf("[ %f, %f ]", v.x, v.y);
	return os;
}

//
// Vector3 Declaration
//

template<typename T>
class Vector3 {
public:
	// Vector3 Public Methods
	Vector3() { x = y = z = 0; }
	Vector3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }

	Vector3<T> operator+(const Vector3<T> &v) const {
		return Vector3(x + v.x, y + v.y, z + v.z);
	}
	Vector3<T> &operator+=(const Vector3<T> &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	Vector3<T> operator-(const Vector3<T> &v) const {
		return Vector3(x - v.x, y - v.y, z - v.z);
	}
	Vector3<T> &operator-=(const Vector3<T> &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }
	bool operator==(const Vector3<T> &v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator!=(const Vector3<T> &v) const { return x != v.x || y != v.y || z != v.z; }

	template<typename U>
	Vector3<T> operator*(U f) const {
		return Vector3<T>(f * x, f * y, f * z);
	}
	template<typename U>
	Vector3<T> &operator*=(U f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	template<typename U>
	Vector3<T> operator/(U f) const {
		float inv = 1.0f / f;
		return Vector3<T>(x * inv, y * inv, z * inv);
	}
	template<typename U>
	Vector3<T> &operator/=(U f) {
		float inv = 1.0f / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}

	T operator[](int i) const {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}
	T &operator[](int i) {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}

	float LengthSquared() const { return x * x + y * y + z * z; }
	float Length() const { return std::sqrt(LengthSquared()); }

	// Vector3 Public Data
	T x, y, z;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
	os << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
	return os;
}
template<>
inline std::ostream &operator<<(std::ostream &os, const Vector3<float> &v) {
	os << StringPrintf("[ %f, %f, %f ]", v.x, v.y, v.z);
	return os;
}

//
// Vector4 Declaration
//

template <typename T>
class Vector4 {
public:
	// Vector4 Public Methods
	Vector4() { x = y = z = 0; w = 1; }
	Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
	Vector4(const Vector3<T>& v, T w = 0) : x(v.x), y(v.y), z(v.z), w(w) {}
	bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z) || isNaN(w); }

	T operator[](int i) const {
		if (i == 0) return x;
		if (i == 1) return y;
		if (i == 2) return z;
		return w;
	}
	T &operator[](int i) {
		if (i == 0) return x;
		if (i == 1) return y;
		if (i == 2) return z;
		return w;
	}

	bool operator==(const Vector4<T> &v) const {
		return x == v.x && y == v.y && z == v.z && w == v.w;
	}
	bool operator!=(const Vector3<T> &v) const {
		return x != v.x || y != v.y || z != v.z || w != v.w;
	}

	inline Vector3<T> GetXYZ() { return Vector3<T>(x, y, z); }

	// Vector4 Public Data
	T x, y, z, w;
};

typedef Vector2<float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector2<uint8_t> Vector2u;
typedef Vector2<int64_t> Vector2l;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
typedef Vector3<uint8_t> Vector3u;
typedef Vector3<int64_t> Vector3l;
typedef Vector4<float> Vector4f;
typedef Vector4<int> Vector4i;
typedef Vector4<uint8_t> Vector4u;
typedef Vector4<int64_t> Vector4l;

//
// Bounds2 Declaration
//

template<typename T>
class Bounds2 {
public:
	// Bounds2 Public Methods
	Bounds2() {
		pMin = Vector2<T>(0, 0);
		pMax = Vector2<T>(0, 0);
	}
	Bounds2(const Vector2<T> &p1, const Vector2<T> &p2) {
		pMin = Vector2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
		pMax = Vector2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
	}
	Bounds2(T xx, T yy, T ww, T hh) {
		pMin = Vector2<T>(xx, yy);
		pMax = Vector2<T>(xx + ww, yy + hh);
	}
	explicit Bounds2(const Vector2<T> &p) : pMin(p), pMax(p) {}

	template<typename U>
	explicit operator Bounds2<U>() const {
		return Bounds2<U>((Vector2<U>) pMin, (Vector2<U>) pMax);
	}

	template<typename U>
	Bounds2<T> operator*(U f) const {
		return Bounds2<T>(pMin * f, pMax * f);
	}
	template<typename U>
	Bounds2<T> &operator*=(U f) {
		pMin *= f;
		pMax *= f;
		if (f < 0)
			std::swap(pMin, pMax);
		return *this;
	}
	template<typename U>
	Bounds2<T> operator/(U f) const {
		float inv = 1.0f / f;
		return Bounds2<T>(pMin * inv, pMax * inv);
	}
	template<typename U>
	Bounds2<T> &operator/=(U f) {
		float inv = 1.0f / f;
		pMin *= inv;
		pMax *= inv;
		if(inv < 0)
			std::swap(pMin, pMax);
		return *this;
	}

	inline const Vector2<T> &operator[](int i) const {
		return (i == 0) ? pMin : pMax;
	}
	inline Vector2<T> &operator[](int i) {
		return (i == 0) ? pMin : pMax;
	}

	bool operator==(const Bounds2<T> &b) const {
		return b.pMin == pMin && b.pMax == pMax;
	}
	bool operator!=(const Bounds2<T> &b) const {
		return b.pMin != pMin || b.pMax != pMax;
	}

	Vector2<T> Diagonal() const { return pMax - pMin; }
	T Area() const {
		Vector2<T> d = pMax - pMin;
		return (d.x * d.y);
	}
	int MaximumExtent() const {
		Vector2<T> diag = Diagonal();
		if (diag.x > diag.y)
			return 0;
		else
			return 1;
	}
	Vector2<T> Lerp(const Vector2f &t) const {
		return Vector2<T>(Lerp(t.x, pMin.x, pMax.x),
			Lerp(t.y, pMin.y, pMax.y));
	}
	Vector2<T> Offset(const Vector2<T> &p) const {
		Vector2<T> o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
		return o;
	}

	void BoundingSphere(Vector2<T> *c, float *rad) const {
		*c = (pMin + pMax) / 2;
		*rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
	}

	// Bounds2 Public Data
	Vector2<T> pMin, pMax;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Bounds2<T> &b) {
	os << "[ " << b.pMin << " - " << b.pMax << " ]";
	return os;
}

//
// Bounds3 Declaration
//

template<typename T>
class Bounds3 {
public:
	Bounds3() {
		T minNum = std::numeric_limits<T>::lowest();
		T maxNum = std::numeric_limits<T>::max();
		pMin = Vector3<T>(maxNum, maxNum, maxNum);
		pMax = Vector3<T>(minNum, minNum, minNum);
	}
	Bounds3(const Vector3<T> &p1, const Vector3<T> &p2)
		: pMin(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z)),
		pMax(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z)) {}
	explicit Bounds3(const Vector3<T> &p) : pMin(p), pMax(p) {}

	template<typename U>
	explicit operator Bounds3<U>() const {
		return Bounds3<U>((Vector3<U>) pMin, (Vector3<U>) pMax);
	}

	template<typename U>
	Bounds3<T> operator*(U f) const {
		return Bounds3<T>(pMin * f, pMax * f);
	}
	template<typename U>
	Bounds3<T> &operator*=(U f) {
		pMin *= f;
		pMax *= f;
		if (f < 0)
			std::swap(pMin, pMax);
		return *this;
	}
	template<typename U>
	Bounds3<T> operator/(U f) const {
		float inv = 1.0f / f;
		return Bounds3<T>(pMin * inv, pMax * inv);
	}
	template<typename U>
	Bounds3<T> &operator/=(U f) {
		float inv = 1.0f / f;
		pMin *= inv;
		pMax *= inv;
		if (inv < 0)
			std::swap(pMin, pMax);
		return *this;
	}

	inline const Vector3<T> &operator[](int i) const {
		return (i == 0) ? pMin : pMax;
	}
	inline Vector3<T> &operator[](int i) {
		return (i == 0) ? pMin : pMax;
	}

	bool operator==(const Bounds3<T> &b) const {
		return b.pMin == pMin && b.pMax == pMax;
	}
	bool operator!=(const Bounds3<T> &b) const {
		return b.pMin != pMin || b.pMax != pMax;
	}

	Vector3<T> Corner(int corner) const {
		return Vector3<T>((*this)[(corner & 1)].x,
			(*this)[(corner & 2) ? 1 : 0].y,
			(*this)[(corner & 4) ? 1 : 0].z);
	}
	Vector3<T> Diagonal() const { return pMax - pMin; }
	T SurfaceArea() const {
		Vector3<T> d = Diagonal();
		return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
	}
	T Volume() const {
		Vector3<T> d = Diagonal();
		return d.x * d.y * d.z;
	}
	int MaximumExtent() const {
		Vector3<T> d = Diagonal();
		if (d.x > d.y && d.x > d.z)
			return 0;
		else if (d.y > d.z)
			return 1;
		else
			return 2;
	}
	Vector3<T> Lerp(const Vector3f &t) const {
		return Vector3<T>(Lerp(t.x, pMin.x, pMax.x),
			Lerp(t.y, pMin.y, pMax.y),
			Lerp(t.z, pMin.z, pMax.z));
	}
	Vector3<T> Offset(const Vector3<T> &p) const {
		Vector3<T> o = p - pMin;
		if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
		if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
		if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
		return o;
	}

	void BoundingSphere(Vector3<T> *center, float *radius) const {
		*center = (pMin + pMax) / 2;
		*radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
	}

	// Bounds3 Public Data
	Vector3<T> pMin, pMax;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Bounds3<T> &b) {
	os << "[ " << b.pMin << " - " << b.pMax << " ]";
	return os;
}

typedef Bounds2<float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds2<uint8_t> Bounds2u;
typedef Bounds2<int64_t> Bounds2l;
typedef Bounds3<float> Bounds3f;
typedef Bounds3<int> Bounds3i;
typedef Bounds3<uint8_t> Bounds3u;
typedef Bounds3<int64_t> Bounds3l;

//
// Bounds2iIterator Declaration
//

class Bounds2iIterator : public std::forward_iterator_tag {
public:
	Bounds2iIterator(const Bounds2i &b, const Vector2i &pt)
		: p(pt), bounds(&b) {}

	// Prefix Overload
	Bounds2iIterator operator++() {
		advance();
		return *this;
	}

	// Postfix Overload
	Bounds2iIterator operator++(int) {
		Bounds2iIterator old = *this;
		advance();
		return old;
	}

	bool operator==(const Bounds2iIterator &bi) const {
		return p == bi.p && bounds == bi.bounds;
	}

	bool operator!=(const Bounds2iIterator &bi) const {
		return p != bi.p || bounds != bi.bounds;
	}

	Vector2i operator*() const { return p; }

private:
	void advance() {
		++p.x;
		if (p.x == bounds->pMax.x) {
			p.x = bounds->pMin.x;
			++p.y;
		}
	}

	Vector2i p;
	const Bounds2i *bounds;
};

//
// complex Declaration
//
template<typename T>
class complex {
public:
	// complex Public Methods
	complex() : re(0), im(0) {}
	complex(T r, T i = 0) : re(r), im(i) {}

	template<typename U>
	explicit operator complex<U>() const {
		return complex<U>(u(re), U(im));
	}
	inline complex conj() const {
		return complex<T>(re, -im);
	}

	bool operator==(const complex<T>& c) { return re == c.re && im == c.im; }
	bool operator!=(const complex<T>& c) { return re != c.re || im != c.im; }

	// complex Public Data
	T re, im;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const complex<T> &v) {
	os << v.re << " re, " << v.im << " im";
	return os;
}
template<>
inline std::ostream &operator<<(std::ostream &os, const complex<float> &v) {
	os << StringPrintf("%f re, %f im", v.re, v.im);
	return os;
}

typedef complex<float> complexf;
typedef complex<int> complexi;

//////////////////////////////////////////////////////////////////////
// Geometry Inline Functions
//////////////////////////////////////////////////////////////////////

//
// Vector3
//

template<typename T, typename U>
inline Vector3<T> operator*(U s, const Vector3<T> &v) {
	return v * s;
}

template<typename T>
Vector3<T> Abs(const Vector3<T> &v) {
	return Vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

template<typename T>
inline T Dot(const Vector3<T> &v1, const Vector3<T> &v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T>
inline T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) {
	return std::abs(Dot(v1, v2));
}

template<typename T>
inline Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
	double v1x = v1.x, v1y = v1.y, v1z = v1.z;
	double v2x = v2.x, v2y = v2.y, v2z = v2.z;
	return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
		(v1x * v2y) - (v1y * v2x));
}

template<typename T>
inline Vector3<T> Normalize(const Vector3<T> &v) {
	return v / v.Length();
}

template<typename T>
T MinComponent(const Vector3<T> &v) {
	return std::min(v.x, std::min(v.y, v.z));
}

template<typename T>
T MaxComponent(const Vector3<T> &v) {
	return std::max(v.x, std::max(v.y, v.z));
}

template<typename T>
int MinDimension(const Vector3<T> &v) {
	return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : ((v.y < v.z) ? 1 : 2);
}

template<typename T>
int MaxDimension(const Vector3<T> &v) {
	return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

template<typename T>
Vector3<T> Min(const Vector3<T> &p1, const Vector3<T> &p2) {
	return Vector3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
		std::min(p1.z, p2.z));
}

template<typename T>
Vector3<T> Max(const Vector3<T> &p1, const Vector3<T> &p2) {
	return Vector3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
		std::max(p1.z, p2.z));
}

template<typename T>
inline Vector3<T> Permute(const Vector3<T> &v, int x, int y, int z) {
	return Vector3<T>(v[x], v[y], v[z]);
}

template<typename T>
inline void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2, Vector3<T> *v3) {
	if (std::abs(v1.x) > std::abs(v1.y))
		*v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
	else
		*v2 = Vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
	*v3 = Cross(v1, *v2);
}

template<typename T>
inline float Distance(const Vector3<T> &p1, const Vector3<T> &p2) {
	return (p1 - p2).Length();
}

template<typename T>
inline float DistanceSquared(const Vector3<T> &p1, const Vector3<T> &p2) {
	return (p1 - p2).LengthSquared();
}

template<typename T>
Vector3<T> Lerp(float t, const Vector3<T> &p0, const Vector3<T> &p1) {
	return (1 - t) * p0 + t * p1;
}

template<typename T>
Vector3<T> Floor(const Vector3<T> &p) {
	return Vector3<T>(std::floor(p.x), std::floor(p.y), std::floor(p.z));
}

template<typename T>
Vector3<T> Ceil(const Vector3<T> &p) {
	return Vector3<T>(std::ceil(p.x), std::ceil(p.y), std::ceil(p.z));
}

template<typename T>
inline Vector3<T> Faceforward(const Vector3<T> &n, const Vector3<T> &v) {
	return (Dot(n, v) < 0.f) ? -n : n;
}

//
// Vector2
//

template<typename T>
inline Vector2<T>::Vector2(const Vector3<T> &p)
	: x(p.x), y(p.y) {}

template<typename T, typename U>
inline Vector2<T> operator*(U f, const Vector2<T> &v) {
	return v * f;
}

template<typename T>
inline float Dot(const Vector2<T> &v1, const Vector2<T> &v2) {
	return v1.x * v2.x + v1.y * v2.y;
}

template<typename T>
inline float AbsDot(const Vector2<T> &v1, const Vector2<T> &v2) {
	return std::abs(Dot(v1, v2));
}

template<typename T>
inline Vector2<T> Normalize(const Vector2<T> &v) {
	return v / v.Length();
}

template<typename T>
Vector2<T> Abs(const Vector2<T> &v) {
	return Vector2<T>(std::abs(v.x), std::abs(v.y));
}

template<typename T>
inline float Distance(const Vector2<T> &p1, const Vector2<T> &p2) {
	return (p1 - p2).Length();
}

template<typename T>
inline float DistanceSquared(const Vector2<T> &p1, const Vector2<T> &p2) {
	return (p1 - p2).LengthSquared();
}

template<typename T>
Vector2<T> Floor(const Vector2<T> &p) {
	return Vector2<T>(std::floor(p.x), std::floor(p.y));
}

template<typename T>
Vector2<T> Ceil(const Vector2<T> &p) {
	return Vector2<T>(std::ceil(p.x), std::ceil(p.y));
}

template<typename T>
Vector2<T> Lerp(float t, const Vector2<T> &v0, const Vector2<T> &v1) {
	return (1 - t) * v0 + t * v1;
}

template<typename T>
Vector2<T> Min(const Vector2<T> &pa, const Vector2<T> &pb) {
	return Vector2<T>(std::min(pa.x, pb.x), std::min(pa.y, pb.y));
}

template<typename T>
Vector2<T> Max(const Vector2<T> &pa, const Vector2<T> &pb) {
	return Vector2<T>(std::max(pa.x, pb.x), std::max(pa.y, pb.y));
}

//
// Bounds3
//

template<typename T>
Bounds3<T> Union(const Bounds3<T> &b, const Vector3<T> &p) {
	return Bounds3<T>(
		Vector3<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y),
			std::min(b.pMin.z, p.z)),
		Vector3<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y),
			std::max(b.pMax.z, p.z)));
}

template<typename T>
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
	return Bounds3<T>(Vector3<T>(std::min(b1.pMin.x, b2.pMin.x),
		std::min(b1.pMin.y, b2.pMin.y),
		std::min(b1.pMin.z, b2.pMin.z)),
		Vector3<T>(std::max(b1.pMax.x, b2.pMax.x),
			std::max(b1.pMax.y, b2.pMax.y),
			std::max(b1.pMax.z, b2.pMax.z)));
}

template<typename T>
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
	return Bounds3<T>(Vector3<T>(std::max(b1.pMin.x, b2.pMin.x),
		std::max(b1.pMin.y, b2.pMin.y),
		std::max(b1.pMin.z, b2.pMin.z)),
		Vector3<T>(std::min(b1.pMax.x, b2.pMax.x),
			std::min(b1.pMax.y, b2.pMax.y),
			std::min(b1.pMax.z, b2.pMax.z)));
}

template<typename T>
bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) {
	bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
	bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
	bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
	return (x && y && z);
}

template<typename T>
bool Inside(const Vector3<T> &p, const Bounds3<T> &b) {
	return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
		p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template<typename T>
bool InsideExclusive(const Vector3<T> &p, const Bounds3<T> &b) {
	return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
		p.y < b.pMax.y && p.z >= b.pMin.z && p.z < b.pMax.z);
}

template<typename T, typename U>
inline Bounds3<T> Expand(const Bounds3<T> &b, U delta) {
	return Bounds3<T>(b.pMin - Vector3<T>(delta, delta, delta),
		b.pMax + Vector3<T>(delta, delta, delta));
}

// Minimum squared distance from point to box; returns zero if point is
// inside.
template<typename T, typename U>
inline float DistanceSquared(const Vector3<T> &p, const Bounds3<U> &b) {
	float dx = std::max({ float(0), b.pMin.x - p.x, p.x - b.pMax.x });
	float dy = std::max({ float(0), b.pMin.y - p.y, p.y - b.pMax.y });
	float dz = std::max({ float(0), b.pMin.z - p.z, p.z - b.pMax.z });
	return dx * dx + dy * dy + dz * dz;
}

template<typename T, typename U>
inline float Distance(const Vector3<T> &p, const Bounds3<U> &b) {
	return std::sqrt(DistanceSquared(p, b));
}

//
// Bounds2iIterator
//

inline Bounds2iIterator begin(const Bounds2i &b) {
	return Bounds2iIterator(b, b.pMin);
}

// Note - the end iterator is one past the last valid y value
inline Bounds2iIterator end(const Bounds2i &b) {
	// Normally, the ending point is at the minimum x value and one past
	// the last valid y value.
	Vector2i pEnd(b.pMin.x, b.pMax.y);
	// However, if the bounds are degenerate, override the end point to
	// equal the start point so that any attempt to iterate over the bounds
	// exits out immediately.
	if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y)
		pEnd = b.pMin;
	return Bounds2iIterator(b, pEnd);
}

//
// Bounds2
//

template<typename T>
Bounds2<T> Union(const Bounds2<T> &b, const Vector2<T> &p) {
	Bounds2<T> ret(Vector2<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y)),
		Vector2<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y)));
	return ret;
}

template<typename T>
Bounds2<T> Union(const Bounds2<T> &b, const Bounds2<T> &b2) {
	Bounds2<T> ret(
		Vector2<T>(std::min(b.pMin.x, b2.pMin.x), std::min(b.pMin.y, b2.pMin.y)),
		Vector2<T>(std::max(b.pMax.x, b2.pMax.x),
			std::max(b.pMax.y, b2.pMax.y)));
	return ret;
}

template<typename T>
Bounds2<T> Intersect(const Bounds2<T> &b, const Bounds2<T> &b2) {
	Bounds2<T> ret(
		Vector2<T>(std::max(b.pMin.x, b2.pMin.x), std::max(b.pMin.y, b2.pMin.y)),
		Vector2<T>(std::min(b.pMax.x, b2.pMax.x),
			std::min(b.pMax.y, b2.pMax.y)));
	return ret;
}

template<typename T>
bool Overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) {
	bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
	bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
	return (x && y);
}

template<typename T>
bool Inside(const Vector2<T> &pt, const Bounds2<T> &b) {
	return (pt.x >= b.pMin.x && pt.x <= b.pMax.x && pt.y >= b.pMin.y &&
		pt.y <= b.pMax.y);
}

template<typename T>
bool InsideExclusive(const Vector2<T> &pt, const Bounds2<T> &b) {
	return (pt.x >= b.pMin.x && pt.x < b.pMax.x && pt.y >= b.pMin.y &&
		pt.y < b.pMax.y);
}

template<typename T, typename U>
Bounds2<T> Expand(const Bounds2<T> &b, U delta) {
	return Bounds2<T>(b.pMin - Vector2<T>(delta, delta),
		b.pMax + Vector2<T>(delta, delta));
}

template<typename T>
float JaccardDistance(const Bounds2<T>& a, const Bounds2<T>& b) {
	T Aa = a.Area();
	T Ab = b.Area();

	if ((Aa + Ab) <= std::numeric_limits<T>::epsilon()) {
		// jaccard_index = 1 -> distance = 0
		return 0.0;
	}
	double Aab = Intersect(a, b).Area();
	// distance = 1 - jaccard_index
	return 1.0 - Aab / (Aa + Ab - Aab);
}

//
// complex
//

template<typename T>
complex<T> operator-(const complex<T> &a) { return complex<T>(-a.re, -a.im); }

template<typename T>
complex<T> operator+(const complex<T> &a, const complex<T> &b) {
	return complex<T>(a.re + b.re, a.im + b.im);
}

template<typename T, typename U>
inline complex<T> operator+(U b, const complex<T> &a) {
	return complex<T>(a.re + b, a.im);
}

template<typename T, typename U>
inline complex<T> operator+(const complex<T> &a, U b) {
	return complex<T>(a.re + b, a.im);
}

template<typename T>
complex<T> &operator+=(complex<T> &a, const complex<T> &b) {
	a.re += b.re;
	a.im += b.im;
	return a;
}

template<typename T, typename U>
complex<T> &operator+=(complex<T> &a, U b) {
	a.re += b;
	return *a;
}

template<typename T>
complex<T> operator-(const complex<T> &a, const complex<T> &b) {
	return complex<T>(a.re - b.re, a.im - b.im);
}

template<typename T, typename U>
inline complex<T> operator-(U b, const complex<T> &a) {
	return complex<T>(b - a.re, -a.im);
}

template<typename T, typename U>
inline complex<T> operator-(const complex<T> &a, U b) {
	return complex<T>(a.re - b, a.im);
}

template<typename T>
complex<T> &operator-=(complex<T> &a, const complex<T> &b) {
	a.re -= b.re;
	a.im -= b.im;
	return a;
}

template<typename T, typename U>
complex<T> &operator-=(complex<T> &a, U b) {
	a.re -= b;
	return a; 
}

template<typename T>
complex<T> operator*(const complex<T> &a, const complex<T> &b) {
	return complex<T>(a.re*b.re - a.im * b.im, a.re*b.im + a.im * b.re);
}

template<typename T, typename U>
inline complex<T> operator*(U s, const complex<T> &c) {
	return complex<T>(c.re*s, c.im*s);
}

template<typename T, typename U>
inline complex<T> operator*(const complex<T> &c, U s) {
	return complex<T>(c.re*s, c.im*s);
}

template<typename T>
complex<T> &operator*=(complex<T> &a, const complex<T> &b) {
	complex<T> temp = a * b;
	a.re = temp.re;
	a.im = temp.im;
	return a;
}

template<typename T, typename U>
complex<T> &operator*=(complex<T> &a, U b) {
	a.re *= b;
	a.im *= b;
	return a;
}

template<typename T>
complex<T> operator/(const complex<T> &a, const complex<T> &b) {
	float t = 1.0f / ((float)b.re*b.re + (float)b.im*b.im);
	return complex<T>((T)((a.re*b.re + a.im * b.im)*t),
		(T)((-a.re * b.im + a.im * b.re)*t));
}

template<typename T, typename U>
complex<T> operator/(U b, const complex<T> &a) {
	float t = 1.0f / ((float)a.re*a.re + (float)a.im*a.im);
	return complex<T>((T)(b*a.re*t), (T)(-b * a.im *t));
}

template<typename T, typename U>
complex<T> operator/(const complex<T> &a, U b) {
	float t = 1.0f / b;
	return complex<T>(a.re*t, a.im*t);
}

template<typename T>
complex<T> &operator/=(complex<T> &a, const complex<T> &b) {
	complex<T> temp = a / b;
	a.re = temp.re;
	a.im = temp.im;
	return a;
}

template<typename T, typename U>
complex<T> &operator/=(complex<T> &a, U b) {
	float t = 1.0f / b;
	a.re *= t;
	a.im *= t;
	return a;
}

template<typename T> 
inline float abs(const complex<T> &a) {
	return std::sqrt((float)a.re*a.re + (float)a.im*a.im);
}

}	// namespace CSRT

#endif	// CSRT_GEOMETRY_H
