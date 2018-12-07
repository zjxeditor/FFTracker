//
// Filter related operations.
//

#include "Filter.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace CSRT {

using namespace Eigen;

Filter GFilter;

// Filter memory arena management.
static MemoryArena *GFilterArenas = nullptr;

bool Filter::SingleInstance = false;

Filter::Filter() : initialized(false) {
	if (SingleInstance) {
		Critical("Only one \"Filter\" can exists.");
		return;
	}
	SingleInstance = true;
}

Filter::~Filter() {
	if (GFilterArenas) {
		delete[] GFilterArenas;
		GFilterArenas = nullptr;
	}
	SingleInstance = false;
}

void Filter::Initialize() {
	if (initialized) return;
	initialized = true;
	GFilterArenas = new MemoryArena[MaxThreadIndex()];
}

float Filter::ModifiedBessel(int order, float x) {
	// sum m=0:inf 1/(m! * Gamma(m + order + 1)) * (x/2)^(2m + order)
	const float eps = 1e-13f;
	float result = 0.0f;
	float m = 0.0f;
	float gamma = 1.0f;
	for (int i = 2; i <= order; ++i)
		gamma *= i;
	float term = pow(x, order) / (pow(2, order) * gamma);

	while (term > eps * result) {
		result += term;
		// calculate new term in series
		++m;
		term *= (x*x) / (4 * m*(m + order));
	}
	return result;
}

void Filter::ChebWin(MatF& window, int N, float atten) {
	window.Reshape(N, 1);
	float *out = window.Data();

	int nn, i;
	float M, n, sum = 0.0f, max = 0.0f;
	float tg = (float)pow(10.0f, atten / 20.0f);	// 1/r term [2], 10^gamma [2]
	float x0 = cosh(1.0f / (N - 1) * acosh(tg));
	M = (N - 1) / 2.0f;
	if (N % 2 == 0) M = M + 0.5f; // handle even length windows
	for (nn = 0; nn < N / 2 + 1; ++nn) {
		n = nn - M;
		sum = 0.0f;
		for (i = 1; i <= M; ++i) {
			sum += ChebPoly(N - 1, x0*(float)cos(Pi*i / N)) *
				(float)cos(2.0f*n*Pi*i / N);
		}

		out[nn] = tg + 2 * sum;
		out[N - nn - 1] = out[nn];
		if (out[nn] > max)
			max = out[nn];
	}
	for (nn = 0; nn < N; nn++)
		out[nn] /= max;	// normalize
}

void Filter::HannWin(MatF& window, const Vector2i& size) {
	MatF hannRows(size.y, 1, &GFilterArenas[ThreadIndex]);
	MatF hannCols(1, size.x, &GFilterArenas[ThreadIndex]);
	window.Reshape(size.y, size.x);
	float *rowData = hannRows.Data();
	float *colData = hannCols.Data();

	int NN = size.y - 1;
	if (NN != 0) {
		for (int i = 0; i < size.y; ++i)
			rowData[i] = 0.5f * (1.0f - cos(2 * Pi*i / NN));
	} else {
		for (int i = 0; i < size.y; ++i)
			rowData[i] = 1.0f;
	}
	NN = size.x - 1;
	if (NN != 0) {
		for (int i = 0; i < size.x; ++i)
			colData[i] = 1.0f / 2.0f * (1.0f - cos(2 * Pi*i / NN));
	} else {
		for (int i = 0; i < size.x; ++i)
			colData[i] = 1.0f;
	}

	Map<VectorXf> egRows(rowData, size.y);
	Map<RowVectorXf> egCols(colData, size.x);
	Map<MatrixXf> egOut(window.Data(), size.y, size.x);
	egOut.noalias() = egRows * egCols;

	GFilterArenas[ThreadIndex].Reset();
}

void Filter::KaiserWin(MatF& window, const Vector2i& size, float alpha) {
	MatF kaiserRows(size.y, 1, &GFilterArenas[ThreadIndex]);
	MatF kaiserCols(1, size.x, &GFilterArenas[ThreadIndex]);
	window.Reshape(size.y, size.x);
	float *rowData = kaiserRows.Data();
	float *colData = kaiserCols.Data();

	int N = size.y - 1;
	float shape = alpha;
	float den = 1.0 / ModifiedBessel(0, shape);

	for (int n = 0; n <= N; ++n) {
		float K = (2.0f * n * 1.0f / N) - 1.0f;
		float x = sqrt(1.0f - K * K);
		rowData[n] = ModifiedBessel(0, shape * x) * den;
	}

	N = size.x - 1;
	for (int n = 0; n <= N; ++n) {
		float K = (2.0f * n * 1.0f / N) - 1.0f;
		float x = sqrt(1.0f - K * K);
		colData[n] = ModifiedBessel(0, shape * x) * den;
	}

	Map<VectorXf> egRows(rowData, size.y);
	Map<RowVectorXf> egCols(colData, size.x);
	Map<MatrixXf> egOut(window.Data(), size.y, size.x);
	egOut.noalias() = egRows * egCols;

	GFilterArenas[ThreadIndex].Reset();
}

void Filter::ChebyshevWin(MatF& window, const Vector2i& size, float attenuation) {
	MatF cheRow(&GFilterArenas[ThreadIndex]), cheCol(&GFilterArenas[ThreadIndex]);
	ChebWin(cheRow, size.y, attenuation);
	ChebWin(cheCol, size.x, attenuation);

	Map<VectorXf> egRows(cheRow.Data(), size.y);
	Map<RowVectorXf> egCols(cheCol.Data(), size.x);
	Map<MatrixXf> egOut(window.Data(), size.y, size.x);
	egOut.noalias() = egRows * egCols;

	GFilterArenas[ThreadIndex].Reset();
}

void Filter::CircShift(const MatF& src, MatF& dest, int dx, int dy) {
	int rows = src.Rows();
	int cols = src.Cols();
	dest.Reshape(rows, cols, false);
	float *pd = dest.Data();
	const float *ps = src.Data();
	for (int y = 0; y < rows; ++y) {
		int ny = Mod(y + dy, rows);
		for (int x = 0; x < cols; ++x) {
			int nx = Mod(x + dx, cols);
			*(pd + ny * cols + nx) = *(ps++);
		}
	}
}

void Filter::GaussianShapedLabels(MatCF &dest, float sigma, int w, int h) {
	MatF tempM(h, w, &GFilterArenas[ThreadIndex]);
	float w2 = (float)(w / 2);
	float h2 = (float)(h / 2);

	// Calculate for each pixel separatelly.
	float *pt = tempM.Data();
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x)
			*(pt++) = (float)exp(-0.5f / pow(sigma, 2) * (pow(y - h2, 2) + pow(x - w2, 2)));
	}

	// Wrap-arround with the circular shifting.
	MatF shiftedM(&GFilterArenas[ThreadIndex]);
	CircShift(tempM, shiftedM, -(w / 2), -(h / 2));
	GFFT.FFT2(shiftedM, dest);

	GFilterArenas[ThreadIndex].Reset();
}

void Filter::FFT2Collection(const std::vector<MatF> &inMs, std::vector<MatCF> &outMs, MemoryArena *targetArena) {
	int len = (int)inMs.size();
	outMs.clear();
	outMs.resize(len, MatCF(targetArena));
	for (int i = 0; i < len; ++i)
		outMs[i].Reshape(inMs[i].Rows(), inMs[i].Cols(), false);

	ParallelFor([&](int64_t index) {
		GFFT.FFT2(inMs[index], outMs[index]);
	}, len, 4);
}

void Filter::DivideComplexMatrices(const MatCF& A, const MatCF& B, MatCF& dest, ComplexF boff) {
	int rows = A.Rows();
	int cols = A.Cols();
	dest.Reshape(rows, cols, false);

    const ComplexF *pa = A.Data();
    const ComplexF *pb = B.Data();
    ComplexF *pd = dest.Data();
    for (int index = 0; index < A.Size(); ++index)
        *(pd++) = *(pa++) / (*(pb++) + boff);
}

void Filter::MulSpectrums(const MatCF& A, const MatCF& B, MatCF& dest, bool conjB) {
	int rows = A.Rows();
	int cols = A.Cols();
	dest.Reshape(rows, cols, false);

	const ComplexF *pa = A.Data();
	const ComplexF *pb = B.Data();
	ComplexF *pd = dest.Data();

	if(conjB) {
		for (int index = 0; index < A.Size(); ++index)
			*(pd++) = *(pa++) * Conj((*(pb++)));
	} else {
		for (int index = 0; index < A.Size(); ++index)
			*(pd++) = *(pa++) * *(pb++);
	}
}

void Filter::GetSubWindow(const Mat& inM, Mat& outM, const Vector2i& center, int w, int h, Bounds2i* validBounds) {
    int startx = center.x - w / 2;
    int starty = center.y - h / 2;
	Bounds2i roi(startx, starty, w, h);
	int padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
	if (roi.pMin.x < 0) {
		padding_left = -roi.pMin.x;
		roi.pMin.x = 0;
	}
	if (roi.pMin.y < 0) {
		padding_top = -roi.pMin.y;
		roi.pMin.y = 0;
	}
	if (roi.pMax.x > inM.Cols()) {
		padding_right = roi.pMax.x - inM.Cols();
		roi.pMax.x = inM.Cols();
	}
	if (roi.pMax.y > inM.Rows()) {
		padding_bottom = roi.pMax.y - inM.Rows();
		roi.pMax.y = inM.Rows();
	}
	Mat tempM(&GFilterArenas[ThreadIndex]);
	inM.Roi(roi, tempM);
	if (!tempM.MakeBorder(outM, padding_left, padding_top, padding_right, padding_bottom, true))
		outM = tempM;

	if (validBounds != nullptr) {
		auto d = roi.Diagonal();
		*validBounds = Bounds2i(padding_left, padding_top, d.x, d.y);
	}

	GFilterArenas[ThreadIndex].Reset();
}

void Filter::GetSubWindow(const MatF& inM, MatF& outM, const Vector2i& center, int w, int h, Bounds2i* validBounds) {
    int startx = center.x - w / 2;
    int starty = center.y - h / 2;
	Bounds2i roi(startx, starty, w, h);
	int padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
	if (roi.pMin.x < 0) {
		padding_left = -roi.pMin.x;
		roi.pMin.x = 0;
	}
	if (roi.pMin.y < 0) {
		padding_top = -roi.pMin.y;
		roi.pMin.y = 0;
	}
	if (roi.pMax.x > inM.Cols()) {
		padding_right = roi.pMax.x - inM.Cols();
		roi.pMax.x = inM.Cols();
	}
	if (roi.pMax.y > inM.Rows()) {
		padding_bottom = roi.pMax.y - inM.Rows();
		roi.pMax.y = inM.Rows();
	}
	MatF tempM(&GFilterArenas[ThreadIndex]);
	inM.Roi(roi, tempM);
	if (!tempM.MakeBorder(outM, padding_left, padding_top, padding_right, padding_bottom, true))
		outM = tempM;

	if (validBounds != nullptr) {
		auto d = roi.Diagonal();
		*validBounds = Bounds2i(padding_left, padding_top, d.x, d.y);
	}

	GFilterArenas[ThreadIndex].Reset();
}

void Filter::GetSubWindow(const MatCF& inM, MatCF& outM, const Vector2i& center, int w, int h, Bounds2i* validBounds) {
    int startx = center.x - w / 2;
    int starty = center.y - h / 2;
	Bounds2i roi(startx, starty, w, h);
	int padding_left = 0, padding_right = 0, padding_top = 0, padding_bottom = 0;
	if (roi.pMin.x < 0) {
		padding_left = -roi.pMin.x;
		roi.pMin.x = 0;
	}
	if (roi.pMin.y < 0) {
		padding_top = -roi.pMin.y;
		roi.pMin.y = 0;
	}
	if (roi.pMax.x > inM.Cols()) {
		padding_right = roi.pMax.x - inM.Cols();
		roi.pMax.x = inM.Cols();
	}
	if (roi.pMax.y > inM.Rows()) {
		padding_bottom = roi.pMax.y - inM.Rows();
		roi.pMax.y = inM.Rows();
	}
	MatCF tempM(&GFilterArenas[ThreadIndex]);
	inM.Roi(roi, tempM);
	if (!tempM.MakeBorder(outM, padding_left, padding_top, padding_right, padding_bottom, true))
		outM = tempM;

	if (validBounds != nullptr) {
		auto d = roi.Diagonal();
		*validBounds = Bounds2i(padding_left, padding_top, d.x, d.y);
	}

	GFilterArenas[ThreadIndex].Reset();
}

float Filter::SubpixelPeak(const MatF& inM, const Vector2i& p, SubpixelDirection direction) {
	int i_p0, i_p_l, i_p_r;	// indexes
	float p0, p_l, p_r;     // values

	if (direction == SubpixelDirection::Vertical) {
		// neighbouring rows
		i_p0 = p.y;
		i_p_l = Mod(i_p0 - 1, inM.Rows());
		i_p_r = Mod(i_p0 + 1, inM.Rows());
		int px = p.x;
		p0 = inM.Get(i_p0, px);
		p_l = inM.Get(i_p_l, px);
		p_r = inM.Get(i_p_r, px);
	} else if (direction == SubpixelDirection::Horizontal) {
		// neighbouring cols
		i_p0 = p.x;
		i_p_l = Mod(i_p0 - 1, inM.Cols());
		i_p_r = Mod(i_p0 + 1, inM.Cols());
		int py = p.y;
		p0 = inM.Get(py, i_p0);
		p_l = inM.Get(py, i_p_l);
		p_r = inM.Get(py, i_p_r);
	} else {
		Critical("Filter::SubpixelPeak: unknown subpixel peak diraction.");
		return 0.0f;
	}
	// Use parabola method.
	float delta = 0.5f * (p_r - p_l) / (2 * p0 - p_r - p_l);
	if (!std::isfinite(delta)) {
		delta = 0.0f;
	}
	return delta;
}

void Filter::Correlation2D(const MatF &inM, const MatF &kernelM, MatF &outM, BorderType borderType) {
	int krows = kernelM.Rows();
	int kcols = kernelM.Cols();
	if(krows % 2 == 0 || kcols % 2 == 0) {
		Critical("Filter::Correlation2D: kernel width and height must be odd.");
		return;
	}
	int halfkr = krows / 2;
	int halfkc = kcols / 2;
	int rows = inM.Rows();
	int cols = inM.Cols();
	int rowbound = rows - 1;
	int colbound = cols - 1;
	outM.Reshape(rows, cols);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC1, inM.Data());
	cv::Mat cvDstM(rows, cols, CV_32FC1, outM.Data());
	cv::Mat cvKernelM(krows, kcols, CV_32FC1, kernelM.Data());
	int border = cv::BORDER_REFLECT;
	switch(borderType){
		case BorderType::Replicate:
			border = cv::BORDER_REPLICATE;
			break;
		case BorderType::Wrap:
			border = cv::BORDER_WRAP;
			break;
		case BorderType::Zero:
			border = cv::BORDER_CONSTANT;
			break;
		default:
			border = cv::BORDER_REFLECT;
			break;
	}
	cv::filter2D(cvOrgM, cvDstM, -1, cvKernelM, cv::Point(-1, -1), 0, border);
#else
	const float *psrc = inM.Data();
	float *pdest = outM.Data();

	switch (borderType) {
	case BorderType::Zero:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const float *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) continue;
						if (idx < 0 || idx >= cols) continue;
						*pdest += (*(pkernel++)) * (*(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Replicate:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const float *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						idy = Clamp(idy, 0, rowbound);
						idx = Clamp(idx, 0, colbound);
						*pdest += (*(pkernel++)) * (*(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Reflect:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const float *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) idy = ReflectIndex(idy, rows);
						if (idx < 0 || idx >= cols) idx = ReflectIndex(idx, cols);
						*pdest += (*(pkernel++)) * (*(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Wrap:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const float *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) idy = WrapIndex(idy, rows);
						if (idx < 0 || idx >= cols) idx = WrapIndex(idx, cols);
						*pdest += (*(pkernel++)) * (*(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	default:
		Critical("Filter::Correlation2D: unknown border type.");
	}
#endif
}

void Filter::Dilate2D(const MatF &inM, const Mat &kernelM, MatF &outM, BorderType borderType) {
	int krows = kernelM.Rows();
	int kcols = kernelM.Cols();
	if (krows % 2 == 0 || kcols % 2 == 0) {
		Critical("Filter::Dilate2D: kernel width and height must be odd.");
		return;
	}
	int halfkr = krows / 2;
	int halfkc = kcols / 2;
	int rows = inM.Rows();
	int cols = inM.Cols();
	int rowbound = rows - 1;
	int colbound = cols - 1;
	outM.Reshape(rows, cols);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC1, inM.Data());
	cv::Mat cvDstM(rows, cols, CV_32FC1, outM.Data());
	cv::Mat cvKernelM(krows, kcols, CV_8UC1, kernelM.Data());
	int border = cv::BORDER_REFLECT;
	switch(borderType){
		case BorderType::Replicate:
			border = cv::BORDER_REPLICATE;
			break;
		case BorderType::Wrap:
			border = cv::BORDER_WRAP;
			break;
		case BorderType::Zero:
			border = cv::BORDER_CONSTANT;
			break;
		default:
			border = cv::BORDER_REFLECT;
			break;
	}
	cv::dilate(cvOrgM, cvDstM, cvKernelM, cv::Point(-1, -1), 1, border);
#else
	const float *psrc = inM.Data();
	float *pdest = outM.Data();

	switch (borderType) {
	case BorderType::Zero:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) continue;
						if (idx < 0 || idx >= cols) continue;
						if (*(pkernel++) == 0) continue;
						*pdest = std::max(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Replicate:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						idy = Clamp(idy, 0, rowbound);
						idx = Clamp(idx, 0, colbound);
						if (*(pkernel++) == 0) continue;
						*pdest = std::max(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Reflect:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) idy = ReflectIndex(idy, rows);
						if (idx < 0 || idx >= cols) idx = ReflectIndex(idx, cols);
						if (*(pkernel++) == 0) continue;
						*pdest = std::max(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Wrap:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) idy = WrapIndex(idy, rows);
						if (idx < 0 || idx >= cols) idx = WrapIndex(idx, cols);
						if (*(pkernel++) == 0) continue;
						*pdest = std::max(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	default:
		Critical("Filter::Dilate2D: unknown border type.");
	}
#endif
}

void Filter::Erode2D(const MatF &inM, const Mat &kernelM, MatF &outM, BorderType borderType) {
	int krows = kernelM.Rows();
	int kcols = kernelM.Cols();
	if (krows % 2 == 0 || kcols % 2 == 0) {
		Critical("Filter::Dilate2D: kernel width and height must be odd.");
		return;
	}
	int halfkr = krows / 2;
	int halfkc = kcols / 2;
	int rows = inM.Rows();
	int cols = inM.Cols();
	int rowbound = rows - 1;
	int colbound = cols - 1;
	outM.Reshape(rows, cols);

#ifdef WITH_OPENCV
	cv::Mat cvOrgM(rows, cols, CV_32FC1, inM.Data());
	cv::Mat cvDstM(rows, cols, CV_32FC1, outM.Data());
	cv::Mat cvKernelM(krows, kcols, CV_8UC1, kernelM.Data());
	int border = cv::BORDER_REFLECT;
	switch(borderType){
		case BorderType::Replicate:
			border = cv::BORDER_REPLICATE;
			break;
		case BorderType::Wrap:
			border = cv::BORDER_WRAP;
			break;
		case BorderType::Zero:
			border = cv::BORDER_CONSTANT;
			break;
		default:
			border = cv::BORDER_REFLECT;
			break;
	}
	cv::erode(cvOrgM, cvDstM, cvKernelM, cv::Point(-1, -1), 1, border);
#else
	const float *psrc = inM.Data();
	float *pdest = outM.Data();

	switch (borderType) {
	case BorderType::Zero:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) continue;
						if (idx < 0 || idx >= cols) continue;
						if (*(pkernel++) == 0) continue;
						*pdest = std::min(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Replicate:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						idy = Clamp(idy, 0, rowbound);
						idx = Clamp(idx, 0, colbound);
						if (*(pkernel++) == 0) continue;
						*pdest = std::min(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Reflect:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) idy = ReflectIndex(idy, rows);
						if (idx < 0 || idx >= cols) idx = ReflectIndex(idx, cols);
						if (*(pkernel++) == 0) continue;
						*pdest = std::min(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	case BorderType::Wrap:
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				const uint8_t *pkernel = kernelM.Data();
				for (int ky = -halfkr; ky <= halfkr; ++ky) {
					for (int kx = -halfkc; kx <= halfkc; ++kx) {
						int idy = y + ky;
						int idx = x + kx;
						if (idy < 0 || idy >= rows) idy = WrapIndex(idy, rows);
						if (idx < 0 || idx >= cols) idx = WrapIndex(idx, cols);
						if (*(pkernel++) == 0) continue;
						*pdest = std::min(*pdest, *(psrc + idy * cols + idx));
					}
				}
				++pdest;
			}
		}
		break;
	default:
		Critical("Filter::Erode2D: unknown border type.");
	}
#endif
}

}	// namespace CSRT
