//
//  Provide fast fft and inv fft operations with the help of fftw.
//

#include "FFT.h"
#include "Parallel.h"
#include "Memory.h"

namespace CSRT {

FFT GFFT;

// FFT temporary arena. Each threads acces it own arena. Automatically reseted.
static MemoryArena *GFFTArenas = nullptr;

bool FFT::SingleInstance = false;

FFT::FFT() : initialized(false) {
	if(SingleInstance) {
		Critical("Only one instance of \"FFT\" class can exists.");
		return;
	}
	SingleInstance = true;
}

FFT::~FFT() {
	if (!planCache.empty()) {
		for (auto &element : planCache)
			fftwf_destroy_plan(element.second);
		planCache.clear();
	}
	fftwf_cleanup();
	if (GFFTArenas) {
		delete[] GFFTArenas;
		GFFTArenas = nullptr;
	}
	SingleInstance = false;
}

void FFT::Initialize() {
	if (initialized) return;
	initialized = true;
	GFFTArenas = new MemoryArena[MaxThreadIndex()];
}

void FFT::FFT1(const float *in, ComplexF *out, int len) {
	ComplexF *inC = (ComplexF*)GFFTArenas[ThreadIndex].Alloc<TempInit>(len, false);
	memset(inC, 0, len * sizeof(ComplexF));
	float *temp = (float*)inC;
	for (int i = 0; i < len; ++i) {
		*temp = *(in++);
		temp += 2;
	}

	FFT1(inC, out, len);

	GFFTArenas[ThreadIndex].Reset();
}

void FFT::FFT1(const ComplexF *in, ComplexF *out, int len) {
	fftwf_plan plan;
	std::string key = StringPrintf("1f%d%d%d", len, fftwf_alignment_of((float*)in), fftwf_alignment_of((float*)out));
	bool flag = false;

	// Create plan or reuse an existing plan.
	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
		flag = true;
	} else {
#ifdef MEASURE_BEST_FFT
		MatCF tempIn(1, len);
		MatCF tempOut(1, len);
		if (fftwf_alignment_of((float*)in) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)out) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::FFT1: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_dft_1d(len, fftw_cast(in),
				fftw_cast(out), FFTW_FORWARD, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_dft_1d(len, fftw_cast(tempIn.Data()),
				fftw_cast(tempOut.Data()), FFTW_FORWARD, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_dft_1d(len, fftw_cast(in),
			fftw_cast(out), FFTW_FORWARD, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
		}
	FFTWMutex.unlock();

	// Compute forward DFT.
#ifdef MEASURE_BEST_FFT
	fftwf_execute_dft(plan, fftw_cast(in), fftw_cast(out));
#else
	if (flag)
		fftwf_execute_dft(plan, fftw_cast(in), fftw_cast(out));
	else
		fftwf_execute(plan);
#endif
	}

void FFT::FFTInv1(const ComplexF *in, float *out, int len, bool scale) {
	ComplexF *outC = (ComplexF*)GFFTArenas[ThreadIndex].Alloc<TempInit>(len, false);

	FFTInv1(in, outC, false);

	float *temp = (float*)outC;
	float *pdest = out;
	for (int i = 0; i < len; ++i) {
		*(pdest++) = *temp;
		temp += 2;
	}
	if (scale) {
		for (int i = 0; i < len; ++i)
			*(out++) /= len;
	}

	GFFTArenas[ThreadIndex].Reset();
}

void FFT::FFTInv1(const ComplexF *in, ComplexF *out, int len, bool scale) {
	fftwf_plan plan;
	std::string key = StringPrintf("1b%d%d%d", len, fftwf_alignment_of((float*)in), fftwf_alignment_of((float*)out));
	bool flag = false;

	// Create plan or reuse an existing plan.
	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
		flag = true;
	} else {
#ifdef MEASURE_BEST_FFT
		MatCF tempIn(1, len);
		MatCF tempOut(1, len);
		if (fftwf_alignment_of((float*)in) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)out) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::FFTInv1: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_dft_1d(len, fftw_cast(in),
				fftw_cast(out), FFTW_BACKWARD, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_dft_1d(len, fftw_cast(tempIn.Data()),
				fftw_cast(tempOut.Data()), FFTW_BACKWARD, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_dft_1d(len, fftw_cast(in),
			fftw_cast(out), FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
	}
	FFTWMutex.unlock();

	// Compute backward DFT
#ifdef MEASURE_BEST_FFT
	fftwf_execute_dft(plan, fftw_cast(in), fftw_cast(out));
#else
	if (flag)
		fftwf_execute_dft(plan, fftw_cast(in), fftw_cast(out));
	else
		fftwf_execute(plan);
#endif

	if (scale) {
		float *temp = (float*)out;
		int dlen = len * 2;
		for (int i = 0; i < dlen; ++i)
			temp[i] /= len;
	}
}

void FFT::FFT2(const MatF &inM, MatCF &outM) {
	MatCF inMC(&GFFTArenas[ThreadIndex]);
	inM.ToMatCF(inMC);
	FFT2(inMC, outM);
	GFFTArenas[ThreadIndex].Reset();
}

void FFT::FFT2(const MatCF &inM, MatCF &outM) {
	int rows = inM.Rows();
	int cols = inM.Cols();
	outM.Reshape(rows, cols, false);

	fftwf_plan plan;
	std::string key = StringPrintf("f%d%d%d%d", rows, cols, fftwf_alignment_of((float*)inM.Data()), fftwf_alignment_of((float*)outM.Data()));
	bool flag = false;

	// Create plan or reuse an existing plan.
	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
		flag = true;
	} else {
#ifdef MEASURE_BEST_FFT
		MatCF tempIn(rows, cols);
		MatCF tempOut(rows, cols);
		if (fftwf_alignment_of((float*)inM.Data()) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)outM.Data()) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::FFT2: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_dft_2d(rows, cols, fftw_cast(inM.Data()),
				fftw_cast(outM.Data()), FFTW_FORWARD, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_dft_2d(rows, cols, fftw_cast(tempIn.Data()),
				fftw_cast(tempOut.Data()), FFTW_FORWARD, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_dft_2d(rows, cols, fftw_cast(inM.Data()),
			fftw_cast(outM.Data()), FFTW_FORWARD, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
	}
	FFTWMutex.unlock();

	// Compute forward DFT.
#ifdef MEASURE_BEST_FFT
	fftwf_execute_dft(plan, fftw_cast(inM.Data()), fftw_cast(outM.Data()));
#else
	if (flag)
		fftwf_execute_dft(plan, fftw_cast(inM.Data()), fftw_cast(outM.Data()));
	else
		fftwf_execute(plan);
#endif
}

void FFT::FFTInv2(const MatCF &inM, MatF &outM, bool scale) {
	MatCF outMC(&GFFTArenas[ThreadIndex]);
	FFTInv2(inM, outMC, false);
	outMC.ToMatF(outM, ComplexMode::Real);
	if (scale)
		outM.ReValue(1.0f / outM.Size(), 0.0f);
	GFFTArenas[ThreadIndex].Reset();
}

void FFT::FFTInv2(const MatCF &inM, MatCF &outM, bool scale) {
	int rows = inM.Rows();
	int cols = inM.Cols();
	outM.Reshape(rows, cols, false);

	fftwf_plan plan;
	std::string key = StringPrintf("b%d%d%d%d", rows, cols, fftwf_alignment_of((float*)inM.Data()), fftwf_alignment_of((float*)outM.Data()));
	bool flag = false;

	// Create plan or reuse an existing plan.
	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
		flag = true;
	} else {
#ifdef MEASURE_BEST_FFT
		MatCF tempIn(rows, cols);
		MatCF tempOut(rows, cols);
		if (fftwf_alignment_of((float*)inM.Data()) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)outM.Data()) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::FFTInv2: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_dft_2d(rows, cols, fftw_cast(inM.Data()),
				fftw_cast(outM.Data()), FFTW_BACKWARD, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_dft_2d(rows, cols, fftw_cast(tempIn.Data()),
				fftw_cast(tempOut.Data()), FFTW_BACKWARD, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_dft_2d(rows, cols, fftw_cast(inM.Data()),
			fftw_cast(outM.Data()), FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
	}
	FFTWMutex.unlock();

	// Compute backward DFT
#ifdef MEASURE_BEST_FFT
	fftwf_execute_dft(plan, fftw_cast(inM.Data()), fftw_cast(outM.Data()));
#else
	if (flag)
		fftwf_execute_dft(plan, fftw_cast(inM.Data()), fftw_cast(outM.Data()));
	else
		fftwf_execute(plan);
#endif

	if (scale)
		outM.ScaleValue(1.0f / outM.Size());
}

void FFT::FFT2Row(const MatF &inM, MatCF &outM) {
	MatCF inMC(&GFFTArenas[ThreadIndex]);
	inM.ToMatCF(inMC);
	FFT2Row(inMC, outM);
	GFFTArenas[ThreadIndex].Reset();
}

void FFT::FFT2Row(const MatCF &inM, MatCF &outM) {
	int rows = inM.Rows();
	int cols = inM.Cols();
	outM.Reshape(rows, cols, false);

	fftwf_plan plan;
	std::string key = StringPrintf("1f%d%d%d", cols, fftwf_alignment_of((float*)inM.Data()), fftwf_alignment_of((float*)outM.Data()));

	// Create plan or reuse an existing plan.
	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
	} else {
#ifdef MEASURE_BEST_FFT
		MatCF tempIn(1, cols);
		MatCF tempOut(1, cols);
		if (fftwf_alignment_of((float*)inM.Data()) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)outM.Data()) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::FFT2Row: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_dft_1d(cols, fftw_cast(inM.Data()),
				fftw_cast(outM.Data()), FFTW_FORWARD, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_dft_1d(cols, fftw_cast(tempIn.Data()),
				fftw_cast(tempOut.Data()), FFTW_FORWARD, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_dft_1d(cols, fftw_cast(inM.Data()),
			fftw_cast(outM.Data()), FFTW_FORWARD, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
	}
	FFTWMutex.unlock();

	// Compute forward DFT.
	const ComplexF *ps = inM.Data();
	ComplexF *pd = outM.Data();
	for (int y = 0; y < rows; ++y) {
		fftwf_execute_dft(plan, fftw_cast(ps), fftw_cast(pd));
		ps += cols;
		pd += cols;
	}
}

void FFT::FFTInv2Row(const MatCF &inM, MatF &outM, bool scale) {
	MatCF outMC(&GFFTArenas[ThreadIndex]);
	FFTInv2Row(inM, outMC, false);
	outMC.ToMatF(outM, ComplexMode::Real);
	if (scale)
		outM.ReValue(1.0f / outM.Cols(), 0.0f);
	GFFTArenas[ThreadIndex].Reset();
}

void FFT::FFTInv2Row(const MatCF &inM, MatCF &outM, bool scale) {
	int rows = inM.Rows();
	int cols = inM.Cols();
	outM.Reshape(rows, cols, false);

	fftwf_plan plan;
	std::string key = StringPrintf("1b%d%d%d", cols, fftwf_alignment_of((float*)inM.Data()), fftwf_alignment_of((float*)outM.Data()));

	// Create plan or reuse an existing plan.
	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
	} else {
#ifdef MEASURE_BEST_FFT
		MatCF tempIn(1, cols);
		MatCF tempOut(1, cols);
		if (fftwf_alignment_of((float*)inM.Data()) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)outM.Data()) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::FFTInv2Row: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_dft_1d(cols, fftw_cast(inM.Data()),
				fftw_cast(outM.Data()), FFTW_BACKWARD, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_dft_1d(cols, fftw_cast(tempIn.Data()),
				fftw_cast(tempOut.Data()), FFTW_BACKWARD, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_dft_1d(cols, fftw_cast(inM.Data()),
			fftw_cast(outM.Data()), FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
	}
	FFTWMutex.unlock();

	// Compute backword DFT.
	const ComplexF *ps = inM.Data();
	ComplexF *pd = outM.Data();
	for (int y = 0; y < rows; ++y) {
		fftwf_execute_dft(plan, fftw_cast(ps), fftw_cast(pd));
		ps += cols;
		pd += cols;
	}

	if (scale)
		outM.ScaleValue(1.0f / outM.Cols());
}

void FFT::Transpose(const MatF &inM, MatF &outM) {
	int rows = inM.Rows();
	int cols = inM.Cols();
	outM.Reshape(cols, rows, false);
	Transpose(inM.Data(), outM.Data(), rows, cols);
}

void FFT::Transpose(float *in, float *out, int rows, int cols) {
	fftwf_plan plan;
	std::string key = StringPrintf("t%d%d%d%d", rows, cols, fftwf_alignment_of(in), fftwf_alignment_of(out));
	bool flag = false;

	FFTWMutex.lock();
	if (planCache.find(key) != planCache.end()) {
		plan = planCache[key];
		flag = true;
	} else {
		fftwf_iodim howmany_dims[2];
		howmany_dims[0].n = rows;
		howmany_dims[0].is = cols;
		howmany_dims[0].os = 1;
		howmany_dims[1].n = cols;
		howmany_dims[1].is = 1;
		howmany_dims[1].os = rows;
#ifdef MEASURE_BEST_FFT
		MatF tempIn(rows, cols);
		MatF tempOut(rows, cols);
		if (fftwf_alignment_of((float*)in) != fftwf_alignment_of((float*)tempIn.Data()) ||
			fftwf_alignment_of((float*)out) != fftwf_alignment_of((float*)tempOut.Data())) {
			Warning("FFT::Transpose: alignment of temp memory for measure not match input. Fall back to estimate method.");
			plan = fftwf_plan_guru_r2r(0, nullptr, 2, howmany_dims,
				in, out, nullptr, FFTW_ESTIMATE);
		} else {
			plan = fftwf_plan_guru_r2r(0, nullptr, 2, howmany_dims,
				tempIn.Data(), tempOut.Data(), nullptr, FFTW_MEASURE);
		}
#else
		plan = fftwf_plan_guru_r2r(0, nullptr, 2, howmany_dims,
			in, out, nullptr, FFTW_ESTIMATE);
#endif
		planCache[key] = plan;
	}
	FFTWMutex.unlock();

	// Compute transpose.
#ifdef MEASURE_BEST_FFT
	fftwf_execute_r2r(plan, in, out);
#else
	if (flag)
		fftwf_execute_r2r(plan, in, out);
	else
		fftwf_execute(plan);
#endif
}

}	// namespace CSRT
