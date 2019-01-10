////
//// CSRDCF Tracker.
////
//
//#include "Tracker.h"
//#include "DSST.h"
//#include "FeaturesExtractor.h"
//#include "Filter.h"
//#include "../Utility/Parallel.h"
//#include "../Utility/Memory.h"
//#include "../Utility/FFT.h"
//
//namespace CSRT {
//
////
//// Tracker Implementation
////
//
//Tracker::Tracker(bool channelWeights, int pca, int iter, float learnRateOfChannel, float learnRateOfFilter) :
//	initialized(false), useChannelWeights(channelWeights), pcaCount(pca), iterations(iter),
//	weightsLearnRate(learnRateOfChannel), filterLearnRate(learnRateOfFilter),
//	arenas(nullptr), backgroundUpdateFlag(false), targetArea(0.0f), k(0.0f) {}
//
//Tracker::~Tracker() {
//	if (arenas) {
//		delete[] arenas;
//		arenas = nullptr;
//	}
//}
//
//void Tracker::CreateFilters(const std::vector<MatCF>& feats, const MatCF& Y, const MatF& P, std::vector<MatCF>& filters, MemoryArena *targetArena) const {
//	int len = (int)feats.size();
//	filters.clear();
//	filters.resize(len, MatCF(targetArena));
//	for (int i = 0; i < len; ++i)
//		filters[i].Reshape(Y.Rows(), Y.Cols(), false);
//
//	ParallelFor([&](int64_t index) {
//		float mu = 5.0f;
//		float beta = 3.0f;
//		float mu_max = 20.0f;
//		float lambda = mu / 100.0f;
//
//		const MatCF &F = feats[index];
//		MatCF &H = filters[index];
//		MatCF Sxy(&arenas[ThreadIndex]), Sxx(&arenas[ThreadIndex]);
//		MatF h(&arenas[ThreadIndex]);
//		// Initialize.
//		GFilter.MulSpectrums(F, Y, Sxy, true);	// Sxy = F * Y.
//		GFilter.MulSpectrums(F, F, Sxx, true);	// Sxx = F * F.
//		// H = Sxy / [Sxx + lambda].
//		GFilter.DivideComplexMatrices(Sxy, Sxx, H, ComplexF(lambda, lambda));
//		GFFT.FFTInv2(H, h, true);
//		h.CMul(P);
//		GFFT.FFT2(h, H);
//		MatCF L(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
//		MatCF G(H.Rows(), H.Cols(), &arenas[ThreadIndex]);
//
//		for (int iteration = 0; iteration < iterations - 1; ++iteration) {
//			// G = (Sxy + mu * H - L) / (Sxx + mu).
//			CalcGMHelper(Sxy, Sxx, H, L, G, mu);
//			// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
//			CalcHMHelper(G, L, P, H, h, mu, lambda);
//			// Update variables for next iteration.
//			// L = L + mu * (G - H).
//			CalcLMHelper(G, H, L, mu);
//			mu = std::min(mu_max, beta*mu);
//		}
//		// G = (Sxy + mu * H - L) / (Sxx + mu).
//		CalcGMHelper(Sxy, Sxx, H, L, G, mu);
//		// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
//		CalcHMHelper(G, L, P, H, h, mu, lambda);
//	}, len, 1);
//}
//
//void Tracker::CreateFilters1(const std::vector<MatCF>& feats, const std::vector<MatCF> &HTS, const MatCF& Y, const MatF& P, std::vector<MatCF>& filters, MemoryArena *targetArena) const {
//    int len = (int)feats.size();
//    filters.clear();
//    filters.resize(len, MatCF(targetArena));
//    for (int i = 0; i < len; ++i)
//        filters[i].Reshape(Y.Rows(), Y.Cols(), false);
//
//    ParallelFor([&](int64_t index) {
//        float mu = 5.0f;
//        float mu1 = 5.0f;
//        float beta = 3.0f;
//        float beta1 = 3.0f;
//        float mu_max = 20.0f;
//        float mu_max1 = 20.0f;
//        float lambda = mu / 100.0f;
//
//        const MatCF &F = feats[index];
//        MatCF &H = filters[index];
//        MatCF Sxy(&arenas[ThreadIndex]), Sxx(&arenas[ThreadIndex]);
//        MatF h(&arenas[ThreadIndex]);
//        // Initialize.
//        GFilter.MulSpectrums(F, Y, Sxy, true);	// Sxy = F * Y.
//        GFilter.MulSpectrums(F, F, Sxx, true);	// Sxx = F * F.
//        // H = Sxy / [Sxx + lambda].
//        GFilter.DivideComplexMatrices(Sxy, Sxx, H, ComplexF(lambda, lambda));
//        GFFT.FFTInv2(H, h, true);
//        h.CMul(P);
//        GFFT.FFT2(h, H);
//        MatCF L(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
//        MatCF L1(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
//        MatCF G(H.Rows(), H.Cols(), &arenas[ThreadIndex]);
//
//        for (int iteration = 0; iteration < iterations - 1; ++iteration) {
//            // G = (Sxy + mu * H - L) / (Sxx + mu).
//            CalcGMHelper(Sxy, Sxx, H, L, G, mu);
//            // H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).
//            CalcHMHelper1(G, L, HTS[index], L1, P, H, h, mu, mu1, lambda);
//            // Update variables for next iteration.
//            // L = L + mu * (G - H).
//            CalcLMHelper(G, H, L, mu);
//            // L1 = L1 + mu1 * (HT - H).
//            CalcLMHelper(HTS[index], H, L1, mu1);
//
//            mu = std::min(mu_max, beta*mu);
//            mu1 = std::min(mu_max1, beta1*mu1);
//        }
//        // G = (Sxy + mu * H - L) / (Sxx + mu).
//        CalcGMHelper(Sxy, Sxx, H, L, G, mu);
//        // H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).
//        CalcHMHelper1(G, L, HTS[index], L1, P, H, h, mu, mu1, lambda);
//    }, len, 1);
//}
//
//void Tracker::CalcGMHelper(const MatCF& Sxy, const MatCF& Sxx, const MatCF& H, const MatCF& L, MatCF& G, float mu) const {
//	const ComplexF *psxy = Sxy.Data();
//	const ComplexF *psxx = Sxx.Data();
//	const ComplexF *ph = H.Data();
//	const ComplexF *pl = L.Data();
//	ComplexF *pg = G.Data();
//
//	// G = (Sxy + mu * H - L) / (Sxx + mu).
//	for (int i = 0; i < G.Size(); ++i)
//		*(pg++) = (*(psxy++) + mu * (*(ph++)) - *(pl++)) / (*(psxx++) + mu);
//}
//
//void Tracker::CalcHMHelper(const MatCF& G, const MatCF& L, const MatF& P, MatCF& H, MatF& h, float mu, float lambda) const {
//	const ComplexF *pG = G.Data();
//	const ComplexF *pL = L.Data();
//	const float *pP = P.Data();
//	ComplexF *pH = H.Data();
//	float *ph = h.Data();
//
//	// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
//
//	// H = mu * G + L.
//	for (int i = 0; i < H.Size(); ++i)
//		*(pH++) = mu * (*(pG++)) + *(pL++);
//
//	// h = FFTInv2(H).
//	GFFT.FFTInv2(H, h, true);
//
//	// h = h * lm * P.
//	float lm = 1.0f / (lambda + mu);
//	for (int i = 0; i < h.Size(); ++i)
//		*(ph++) *= (lm * (*(pP++)));
//
//	// H = FFT2(h).
//	GFFT.FFT2(h, H);
//}
//
//void Tracker::CalcHMHelper1(const MatCF &G, const MatCF &L, const MatCF &HT, const MatCF &L1, const MatF &P, MatCF &H, MatF &h, float mu, float mu1, float lambda) const {
//    const ComplexF *pG = G.Data();
//    const ComplexF *pL = L.Data();
//    const ComplexF *pHT = HT.Data();
//    const ComplexF *pL1 = L1.Data();
//    const float *pP = P.Data();
//    ComplexF *pH = H.Data();
//    float *ph = h.Data();
//
//    // H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).
//
//    // H = mu * G + L + mu1 * HT + L1.
//    for (int i = 0; i < H.Size(); ++i)
//        *(pH++) = mu * (*(pG++)) + *(pL++) + mu1 * (*(pHT++)) + *(pL1++);
//
//    // h = FFTInv2(H).
//    GFFT.FFTInv2(H, h, true);
//
//    // h = h * lm * P.
//    float lm = 1.0f / (lambda + mu + mu1);
//    for (int i = 0; i < h.Size(); ++i)
//        *(ph++) *= (lm * (*(pP++)));
//
//    // H = FFT2(h).
//    GFFT.FFT2(h, H);
//}
//
//void Tracker::CalcLMHelper(const MatCF& G, const MatCF& H, MatCF& L, float mu) const {
//	const ComplexF *pg = G.Data();
//	const ComplexF *ph = H.Data();
//	ComplexF *pl = L.Data();
//
//	// L = L + mu * (G - H).
//	for (int i = 0; i < L.Size(); ++i)
//		*(pl++) += (mu * (*(pg++) - *(ph++)));
//}
//
//void Tracker::CalculateResponse(const std::vector<MatF>& ftrs, const std::vector<MatCF>& filter, MatF& response) const {
//	std::vector<MatCF> Ffeatures;
//	if (pcaCount > 0) {
//		int ftrNum = (int)ftrs.size();
//		int sampleNum = ftrs[0].Size();
//		MatF mergedFtrs(&arenas[ThreadIndex]);
//		MatF pmergedFtrs(ftrs[0].Rows(), ftrs[0].Cols() * pcaCount, &arenas[ThreadIndex]);
//		mergedFtrs.Merge(ftrs);
//		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
//		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
//		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
//		egProjDataM.noalias() = egDataM * egProjM;
//		std::vector<MatF> pftrs;
//		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
//		GFilter.FFT2Collection(pftrs, Ffeatures, &arenas[ThreadIndex]);
//	} else {
//		GFilter.FFT2Collection(ftrs, Ffeatures, &arenas[ThreadIndex]);
//	}
//	if (Ffeatures.size() != filter.size()) {
//		Critical("Tracker::CalculateResponse: feature channels not match filter channels.");
//		return;
//	}
//
//	int len = (int)Ffeatures.size();
//	int width = Ffeatures[0].Cols();
//	int height = Ffeatures[0].Rows();
//	int supressRadius = 2;
//
//	if (useChannelWeights) {
//		std::vector<MatCF> respfs(len, MatCF(&arenas[ThreadIndex]));
//		std::vector<MatF> resps(len, MatF(&arenas[ThreadIndex]));
//		std::vector<float> detectWeights(len, 0.0f);
//		float weightSum = 0.0f;
//
//		ParallelFor([&](int64_t k) {
//			GFilter.MulSpectrums(Ffeatures[k], filter[k], respfs[k], true);
//			GFFT.FFTInv2(respfs[k], resps[k]);
//			float maxValue;
//			Vector2i maxLoc;
//			resps[k].MaxLoc(maxValue, maxLoc);
//
//			float subValue;
//			Vector2i subLoc;
//			FindSubPeak(resps[k], maxLoc, subLoc, subValue);
//            detectWeights[k] = std::max(1.0f - (subValue / maxValue), 0.5f) * filterWeights[k];
//
////			for (int y = maxLoc.y - supressRadius; y <= maxLoc.y + supressRadius; ++y) {
////				for (int x = maxLoc.x - supressRadius; x <= maxLoc.x + supressRadius; ++x) {
////					int ny = Mod(y, height);
////					int nx = Mod(x, width);
////					*(resps[k].Data() + width * ny + nx) *= -1.0f;
////				}
////			}
////			float subMaxValue;
////			Vector2i subMaxLoc;
////			resps[k].MaxLoc(subMaxValue, subMaxLoc);
////			detectWeights[k] = std::max(1.0f - (subMaxValue / maxValue), 0.5f) * filterWeights[k];
////			for (int y = maxLoc.y - supressRadius; y <= maxLoc.y + supressRadius; ++y) {
////				for (int x = maxLoc.x - supressRadius; x <= maxLoc.x + supressRadius; ++x) {
////					int ny = Mod(y, height);
////					int nx = Mod(x, width);
////					*(resps[k].Data() + width * ny + nx) *= -1.0f;
////				}
////			}
//		}, len, 2);
//
//		response.Reshape(height, width);
//		for (int i = 0; i < len; ++i) {
//			weightSum += detectWeights[i];
//		}
//		for (int i = 0; i < len; ++i) {
//			detectWeights[i] /= weightSum;
//			response.CAdd(resps[i], detectWeights[i], 0.0f);
//		}
//	} else {
//		MatCF respf(filter[0].Rows(), filter[0].Cols(), &arenas[ThreadIndex]);
//		MatCF tempf(&arenas[ThreadIndex]);
//		for (int i = 0; i < len; ++i) {
//			GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
//			respf.CAdd(tempf, 1.0f, ComplexF(0.0f, 0.0f));
//		}
//		GFFT.FFTInv2(respf, response);
//	}
//
////    int len = (int)Ffeatures.size();
////    MatCF respf(filter[0].Rows(), filter[0].Cols(), &arenas[ThreadIndex]);
////    MatCF tempf(&arenas[ThreadIndex]);
////    if (useChannelWeights) {
////        for (int i = 0; i < len; ++i) {
////            GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
////            respf.CAdd(tempf, filterWeights[i], ComplexF(0.0f, 0.0f));
////        }
////    } else {
////        for (int i = 0; i < len; ++i) {
////            GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
////            respf.CAdd(tempf, 1.0f, ComplexF(0.0f, 0.0f));
////        }
////    }
////    GFFT.FFTInv2(respf, response);
//}
//
//float Tracker::CalculateScore(const MatF &response, const Vector2i &pos, float maxResp) const {
//    const static int offsetX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
//    const static int offsetY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
//
//    const int rows = response.Rows();
//    const int cols = response.Cols();
//
//    float distance = 0.0f;
//    float score = std::numeric_limits<float>::max();
//    bool peakFlag = false;
//    int tx = 0, ty = 0;
//    float cvalue = 0.0f, tscore = 0.0f;
//    for(int r = 0; r < rows; ++r) {
//        for(int c = 0; c < cols; ++c) {
//            if(r == pos.y && c == pos.x) continue;
//            // judge peak
//            peakFlag = true;
//            cvalue = response.Get(r, c);
//            for(int i = 0; i < 8; ++i) {
//                tx = c + offsetX[i];
//                ty = r + offsetY[i];
//                tx = Mod(tx, cols);
//                ty = Mod(ty, rows);
//                if(response.Get(ty, tx) >= cvalue) {
//                    peakFlag = false;
//                    break;
//                }
//            }
//            if(!peakFlag) continue;
//
//            // calculate value
//            distance = std::pow(pos.x - c, 2.0f) + std::pow(pos.y - r, 2.0f);
//            distance = 1.0f - std::exp(-k / 2.0f * distance);
//            tscore = (maxResp - cvalue) / distance;
//            if(tscore < score) score = tscore;
//        }
//    }
//
//    return score;
//}
//
//void Tracker::FindSubPeak(const MatF &response, const Vector2i &peakPos, Vector2i &subPos, float &subValue) const {
//    const static int offsetX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
//    const static int offsetY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
//    const int rows = response.Rows();
//    const int cols = response.Cols();
//    subValue = 0.0f;
//
//    bool peakFlag = false;
//    int tx = 0, ty = 0;
//    float cvalue = 0.0f;
//    for(int r = 0; r < rows; ++r) {
//        for(int c = 0; c < cols; ++c) {
//            if(r == peakPos.y && c == peakPos.x) continue;
//            // judge peak
//            peakFlag = true;
//            cvalue = response.Get(r, c);
//            for(int i = 0; i < 8; ++i) {
//                tx = c + offsetX[i];
//                ty = r + offsetY[i];
//                tx = Mod(tx, cols);
//                ty = Mod(ty, rows);
//                if(response.Get(ty, tx) >= cvalue) {
//                    peakFlag = false;
//                    break;
//                }
//            }
//            if(!peakFlag) continue;
//
//            // calculate value
//            if(cvalue > subValue) {
//                subValue = cvalue;
//                subPos.x = c;
//                subPos.y = r;
//            }
//        }
//    }
//}
//
//void Tracker::ResetArenas(bool background) const {
//	if (!arenas) return;
//	int len = MaxThreadIndex() - 1;
//	if (background) {
//		arenas[len].Reset();
//	} else {
//		for (int i = 0; i < len; ++i)
//			arenas[i].Reset();
//	}
//}
//
////
//// Tracker Main Interface
////
//
//void Tracker::Initialize(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask, float orgArea) {
//	if (initialized) return;
//	initialized = true;
//	if (!arenas)
//		arenas = new MemoryArena[MaxThreadIndex()];
//
//	// Input check.
//	if (feats.size() <= 0 || feats[0].Size() <= 0 || filterMask.Size() <= 0) {
//		Critical("Tracker::Initialize: cannot initialize with empty features data or filterMask.");
//		return;
//	}
//	if (yf.Rows() != filterMask.Rows() || yf.Cols() != filterMask.Cols()) {
//		Critical("Tracker::Initialize: ideal response and filter mask not match.");
//		return;
//	}
//	if(orgArea <= 0.0f) {
//        Critical("Tracker::Initialize: invalid target area.");
//        return;
//	}
//	targetArea = orgArea;
//    k = 8.0f / targetArea;
//
//	// Initialize filter.
//	std::vector<MatCF> Fftrs;
//	if (pcaCount > (int)feats.size()) {
//		Warning("Tracker::Initialize: PCA count is greater than original feature count. Disable PCA.");
//		pcaCount = 0;
//	}
//	if (pcaCount > 0) {
//		// PCA features.
//		int ftrNum = (int)feats.size();
//		int sampleNum = feats[0].Size();
//		MatF mergedFtrs(&arenas[ThreadIndex]);
//		MatF centerdFtrs(sampleNum, ftrNum, &arenas[ThreadIndex]);
//		MatF cov(ftrNum, ftrNum, &arenas[ThreadIndex]);
//		MatF pmergedFtrs(feats[0].Rows(), feats[0].Cols() * pcaCount, &arenas[ThreadIndex]);
//		mergedFtrs.Merge(feats);
//		project.Reshape(ftrNum, pcaCount, false);
//		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
//		Eigen::Map<Eigen::MatrixXf> egCenteredM(centerdFtrs.Data(), sampleNum, ftrNum);
//		Eigen::Map<Eigen::MatrixXf> egCovM(cov.Data(), ftrNum, ftrNum);
//		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
//		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
//		egCenteredM = egDataM.rowwise() - egDataM.colwise().mean();
//		egCovM.noalias() = egCenteredM.transpose() * egCenteredM;
//		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(egCovM);
//		if (eig.info() != Eigen::Success) {
//			Critical("Cannot solve eigen values for PCA process.");
//			return;
//		}
//		egProjM = eig.eigenvectors().rightCols(pcaCount);
//		egProjDataM.noalias() = egDataM * egProjM;
//		std::vector<MatF> pftrs;
//		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
//		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
//	} else {
//		GFilter.FFT2Collection(feats, Fftrs, &arenas[ThreadIndex]);
//	}
//	CreateFilters(Fftrs, yf, filterMask, filters);
//
//	// Initialize channel weights.
//	if (useChannelWeights) {
//		int len = (int)filters.size();
//		filterWeights.resize(len, 0.0f);
//		ParallelFor([&](int64_t index) {
//			MatCF currentRespf(&arenas[ThreadIndex]);
//			MatF currentResp(&arenas[ThreadIndex]);
//			float maxValue = 0.0f;
//			Vector2i maxLoc;
//			GFilter.MulSpectrums(Fftrs[index], filters[index], currentRespf, true);
//			GFFT.FFTInv2(currentRespf, currentResp, true);
//			currentResp.MaxLoc(maxValue, maxLoc);
//			filterWeights[index] = maxValue;
//		}, len, 1);
//		float sumWeights = 0.0f;
//		for (int i = 0; i < len; ++i)
//			sumWeights += filterWeights[i];
//		for (int i = 0; i < len; ++i)
//			filterWeights[i] /= sumWeights;
//	}
//
//	ResetArenas(false);
//	ResetArenas(true);
//}
//
//void Tracker::GetPosition(const std::vector<MatF> &feats, float currentScale, int cellSize, float rescaleRatio,
//	const Vector2f &currentPos, const Vector2i &moveSize, Vector2f &newPos, float &score) const {
//	if (!initialized) {
//		Error("Tracker::GetPosition: tracker is not initialized.");
//		return;
//	}
//
//	MatF response(&arenas[ThreadIndex]);
//	CalculateResponse(feats, filters, response);
//	float maxValue;
//	Vector2i maxLoc;
//	response.MaxLoc(maxValue, maxLoc);
//	score = CalculateScore(response, maxLoc, maxValue);
//
//	// Take into account also subpixel accuracy.
//	float x = maxLoc.x + GFilter.SubpixelPeak(response, maxLoc, SubpixelDirection::Horizontal);
//	float y = maxLoc.y + GFilter.SubpixelPeak(response, maxLoc, SubpixelDirection::Vertical);
//	if (x > (response.Cols() - 1) / 2.0f) x -= response.Cols();
//	if (y > (response.Rows() - 1) / 2.0f) y -= response.Rows();
//
//	// Calculate new position.
//	float scale = currentScale * cellSize / rescaleRatio;
//	newPos = currentPos + Vector2f(scale * x, scale * y);
//
//	// Fit to bound.
//	newPos.x = Clamp(newPos.x, 0.0f, moveSize.x - 1.0f);
//	newPos.y = Clamp(newPos.y, 0.0f, moveSize.y - 1.0f);
//
//	ResetArenas(false);
//}
//
//void Tracker::Update(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask) {
//	if (!initialized) {
//		Error("Tracker::Update: tracker is not initialized.");
//		return;
//	}
//
//	// Input check.
//	if (feats.size() <= 0 || feats[0].Size() <= 0 || filterMask.Size() <= 0) {
//		Critical("Tracker::Initialize: cannot initialize with empty features data or filterMask.");
//		return;
//	}
//	if (yf.Rows() != filterMask.Rows() || yf.Cols() != filterMask.Cols()) {
//		Critical("Tracker::Initialize: ideal response and filter mask not match.");
//		return;
//	}
//
//	std::vector<MatCF> Fftrs;
//	if (pcaCount > 0) {
//		int ftrNum = (int)feats.size();
//		int sampleNum = feats[0].Size();
//		MatF mergedFtrs(&arenas[ThreadIndex]);
//		MatF pmergedFtrs(feats[0].Rows(), feats[0].Cols() * pcaCount, &arenas[ThreadIndex]);
//		mergedFtrs.Merge(feats);
//		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
//		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
//		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
//		egProjDataM.noalias() = egDataM * egProjM;
//		std::vector<MatF> pftrs;
//		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
//		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
//	} else {
//		GFilter.FFT2Collection(feats, Fftrs, &arenas[ThreadIndex]);
//	}
//	std::vector<MatCF> newFilters;
//	CreateFilters(Fftrs, yf, filterMask, newFilters, &arenas[ThreadIndex]);
//
//	// Calculate per channel weights.
//	int len = (int)newFilters.size();
//	if (useChannelWeights) {
//		std::vector<float> newFilterWeights(newFilters.size(), 0.0f);
//		ParallelFor([&](int64_t index) {
//			MatCF currentRespf(&arenas[ThreadIndex]);
//			MatF currentResp(&arenas[ThreadIndex]);
//			float maxValue = 0.0f;
//			Vector2i maxLoc;
//			GFilter.MulSpectrums(Fftrs[index], newFilters[index], currentRespf, true);
//			GFFT.FFTInv2(currentRespf, currentResp, true);
//			currentResp.MaxLoc(maxValue, maxLoc);
//			newFilterWeights[index] = maxValue;
//		}, len, 1);
//		float sumWeights = 0.0f;
//		for (int i = 0; i < len; ++i)
//			sumWeights += newFilterWeights[i];
//
//		// Update filter weights with new values.
//		float sumUpdate = 0.0f;
//		for (int i = 0; i < len; ++i) {
//			filterWeights[i] = Lerp(weightsLearnRate, filterWeights[i], newFilterWeights[i] / sumWeights);
//			sumUpdate += filterWeights[i];
//		}
//		// Normalize weights.
//		for (int i = 0; i < len; ++i)
//			filterWeights[i] /= sumUpdate;
//	}
//
//	// Update filters.
//	ParallelFor([&](int64_t i) {
//		filters[i].Lerp(newFilters[i], filterLearnRate);
//	}, len, 2);
//
//	ResetArenas(false);
//}
//
//void Tracker::StartBackgroundUpdate() {
//	if (backgroundUpdateFlag) return;
//	bk_filters = filters;
//	bk_filterWeights = filterWeights;
//	backgroundUpdateFlag = true;
//}
//
//void Tracker::FetchUpdateResult() {
//	if (!backgroundUpdateFlag) return;
//	filters = bk_filters;
//	filterWeights = bk_filterWeights;
//	backgroundUpdateFlag = false;
//}
//
//void Tracker::BackgroundUpdate(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask) {
//	// Extract features.
//	std::vector<MatCF> Fftrs;
//	if (pcaCount > 0) {
//		int ftrNum = (int)feats.size();
//		int sampleNum = feats[0].Size();
//		MatF mergedFtrs(&arenas[ThreadIndex]);
//		MatF pmergedFtrs(feats[0].Rows(), feats[0].Cols() * pcaCount, &arenas[ThreadIndex]);
//		mergedFtrs.Merge(feats);
//		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
//		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
//		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
//		egProjDataM.noalias() = egDataM * egProjM;
//		std::vector<MatF> pftrs;
//		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
//		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
//	} else {
//		GFilter.FFT2Collection(feats, Fftrs, &arenas[ThreadIndex]);
//	}
//	std::vector<MatCF> newFilters;
//	CreateFilters(Fftrs, yf, filterMask, newFilters, &arenas[ThreadIndex]);
//
//	// Calculate per channel weights.
//	int len = (int)newFilters.size();
//	if (useChannelWeights) {
//		std::vector<float> newFilterWeights(newFilters.size(), 0.0f);
//		ParallelFor([&](int64_t index) {
//			MatCF currentRespf(&arenas[ThreadIndex]);
//			MatF currentResp(&arenas[ThreadIndex]);
//			float maxValue = 0.0f;
//			Vector2i maxLoc;
//			GFilter.MulSpectrums(Fftrs[index], newFilters[index], currentRespf, true);
//			GFFT.FFTInv2(currentRespf, currentResp, true);
//			currentResp.MaxLoc(maxValue, maxLoc);
//			newFilterWeights[index] = maxValue;
//		}, len, 1);
//		float sumWeights = 0.0f;
//		for (int i = 0; i < len; ++i)
//			sumWeights += newFilterWeights[i];
//
//		// Update filter weights with new values.
//		float sumUpdate = 0.0f;
//		for (int i = 0; i < len; ++i) {
//			bk_filterWeights[i] = Lerp(weightsLearnRate, bk_filterWeights[i], newFilterWeights[i] / sumWeights);
//			sumUpdate += bk_filterWeights[i];
//		}
//		// Normalize weights.
//		for (int i = 0; i < len; ++i)
//			bk_filterWeights[i] /= sumUpdate;
//	}
//
//	// Update filters.
//	ParallelFor([&](int64_t i) {
//		bk_filters[i].Lerp(newFilters[i], filterLearnRate);
//	}, len, 2);
//
//	ResetArenas(true);
//}
//
//}	// namespace CSRT

//
// CSRDCF Tracker.
//

#include "Tracker.h"
#include "DSST.h"
#include "FeaturesExtractor.h"
#include "Filter.h"
#include "../Utility/Parallel.h"
#include "../Utility/Memory.h"
#include "../Utility/FFT.h"

namespace CSRT {

//
// Tracker Implementation
//

Tracker::Tracker(bool channelWeights, int pca, int iter, const std::vector<float> &channelRates, const std::vector<float> &filterRates) :
		initialized(false), useChannelWeights(channelWeights), pcaCount(pca), iterations(iter),
		arenas(nullptr), backgroundUpdateFlag(false), targetArea(0.0f), k(0.0f) {
    SetLearningRates(channelRates, filterRates);
}

Tracker::~Tracker() {
	if (arenas) {
		delete[] arenas;
		arenas = nullptr;
	}
}

void Tracker::SetLearningRates(const std::vector<float> &channelRates, const std::vector<float> &filterRates) {
	if(channelRates.size() <= 0 || channelRates.size() != filterRates.size()) {
		Critical("Tracker::SetLearningRates: invalid learning rates configuration.");
	}
	weightsLearnRates = channelRates;
	filterLearnRates = filterRates;
	int len = (int)filterLearnRates.size();
	filters.resize(len);
	filterWeights.resize(len);
	bk_filters.resize(len);
	bk_filterWeights.resize(len);
}

void Tracker::CreateFilters(const std::vector<MatCF>& feats, const MatCF& Y, const MatF& P, std::vector<MatCF>& filters, MemoryArena *targetArena) const {
	int len = (int)feats.size();
	filters.clear();
	filters.resize(len, MatCF(targetArena));
	for (int i = 0; i < len; ++i)
		filters[i].Reshape(Y.Rows(), Y.Cols(), false);

	ParallelFor([&](int64_t index) {
		float mu = 5.0f;
		float beta = 3.0f;
		float mu_max = 20.0f;
		float lambda = mu / 100.0f;

		const MatCF &F = feats[index];
		MatCF &H = filters[index];
		MatCF Sxy(&arenas[ThreadIndex]), Sxx(&arenas[ThreadIndex]);
		MatF h(&arenas[ThreadIndex]);
		// Initialize.
		GFilter.MulSpectrums(F, Y, Sxy, true);	// Sxy = F * Y.
		GFilter.MulSpectrums(F, F, Sxx, true);	// Sxx = F * F.
		// H = Sxy / [Sxx + lambda].
		GFilter.DivideComplexMatrices(Sxy, Sxx, H, ComplexF(lambda, lambda));
		GFFT.FFTInv2(H, h, true);
		h.CMul(P);
		GFFT.FFT2(h, H);
		MatCF L(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
		MatCF G(H.Rows(), H.Cols(), &arenas[ThreadIndex]);

		for (int iteration = 0; iteration < iterations - 1; ++iteration) {
			// G = (Sxy + mu * H - L) / (Sxx + mu).
			CalcGMHelper(Sxy, Sxx, H, L, G, mu);
			// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
			CalcHMHelper(G, L, P, H, h, mu, lambda);
			// Update variables for next iteration.
			// L = L + mu * (G - H).
			CalcLMHelper(G, H, L, mu);
			mu = std::min(mu_max, beta*mu);
		}
		// G = (Sxy + mu * H - L) / (Sxx + mu).
		CalcGMHelper(Sxy, Sxx, H, L, G, mu);
		// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).
		CalcHMHelper(G, L, P, H, h, mu, lambda);
	}, len, 1);
}

void Tracker::CreateFilters1(const std::vector<MatCF>& feats, const std::vector<MatCF> &HTS, const MatCF& Y, const MatF& P, std::vector<MatCF>& filters, MemoryArena *targetArena) const {
	int len = (int)feats.size();
	filters.clear();
	filters.resize(len, MatCF(targetArena));
	for (int i = 0; i < len; ++i)
		filters[i].Reshape(Y.Rows(), Y.Cols(), false);

	ParallelFor([&](int64_t index) {
		float mu = 5.0f;
		float mu1 = 5.0f;
		float beta = 3.0f;
		float beta1 = 3.0f;
		float mu_max = 20.0f;
		float mu_max1 = 20.0f;
		float lambda = mu / 100.0f;

		const MatCF &F = feats[index];
		MatCF &H = filters[index];
		MatCF Sxy(&arenas[ThreadIndex]), Sxx(&arenas[ThreadIndex]);
		MatF h(&arenas[ThreadIndex]);
		// Initialize.
		GFilter.MulSpectrums(F, Y, Sxy, true);	// Sxy = F * Y.
		GFilter.MulSpectrums(F, F, Sxx, true);	// Sxx = F * F.
		// H = Sxy / [Sxx + lambda].
		GFilter.DivideComplexMatrices(Sxy, Sxx, H, ComplexF(lambda, lambda));
		GFFT.FFTInv2(H, h, true);
		h.CMul(P);
		GFFT.FFT2(h, H);
		MatCF L(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
		MatCF L1(H.Rows(), H.Cols(), &arenas[ThreadIndex]);	// Lagrangian multiplier.
		MatCF G(H.Rows(), H.Cols(), &arenas[ThreadIndex]);

		for (int iteration = 0; iteration < iterations - 1; ++iteration) {
			// G = (Sxy + mu * H - L) / (Sxx + mu).
			CalcGMHelper(Sxy, Sxx, H, L, G, mu);
			// H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).
			CalcHMHelper1(G, L, HTS[index], L1, P, H, h, mu, mu1, lambda);
			// Update variables for next iteration.
			// L = L + mu * (G - H).
			CalcLMHelper(G, H, L, mu);
			// L1 = L1 + mu1 * (HT - H).
			CalcLMHelper(HTS[index], H, L1, mu1);

			mu = std::min(mu_max, beta*mu);
			mu1 = std::min(mu_max1, beta1*mu1);
		}
		// G = (Sxy + mu * H - L) / (Sxx + mu).
		CalcGMHelper(Sxy, Sxx, H, L, G, mu);
		// H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).
		CalcHMHelper1(G, L, HTS[index], L1, P, H, h, mu, mu1, lambda);
	}, len, 1);
}

void Tracker::CalcGMHelper(const MatCF& Sxy, const MatCF& Sxx, const MatCF& H, const MatCF& L, MatCF& G, float mu) const {
	const ComplexF *psxy = Sxy.Data();
	const ComplexF *psxx = Sxx.Data();
	const ComplexF *ph = H.Data();
	const ComplexF *pl = L.Data();
	ComplexF *pg = G.Data();

	// G = (Sxy + mu * H - L) / (Sxx + mu).
	for (int i = 0; i < G.Size(); ++i)
		*(pg++) = (*(psxy++) + mu * (*(ph++)) - *(pl++)) / (*(psxx++) + mu);
}

void Tracker::CalcHMHelper(const MatCF& G, const MatCF& L, const MatF& P, MatCF& H, MatF& h, float mu, float lambda) const {
	const ComplexF *pG = G.Data();
	const ComplexF *pL = L.Data();
	const float *pP = P.Data();
	ComplexF *pH = H.Data();
	float *ph = h.Data();

	// H = FFT2(FFTInv2(mu * G + L) / (lambda + mu) * P).

	// H = mu * G + L.
	for (int i = 0; i < H.Size(); ++i)
		*(pH++) = mu * (*(pG++)) + *(pL++);

	// h = FFTInv2(H).
	GFFT.FFTInv2(H, h, true);

	// h = h * lm * P.
	float lm = 1.0f / (lambda + mu);
	for (int i = 0; i < h.Size(); ++i)
		*(ph++) *= (lm * (*(pP++)));

	// H = FFT2(h).
	GFFT.FFT2(h, H);
}

void Tracker::CalcHMHelper1(const MatCF &G, const MatCF &L, const MatCF &HT, const MatCF &L1, const MatF &P, MatCF &H, MatF &h, float mu, float mu1, float lambda) const {
	const ComplexF *pG = G.Data();
	const ComplexF *pL = L.Data();
	const ComplexF *pHT = HT.Data();
	const ComplexF *pL1 = L1.Data();
	const float *pP = P.Data();
	ComplexF *pH = H.Data();
	float *ph = h.Data();

	// H = FFT2(FFTInv2(mu * G + L + mu1 * HT + L1) / (lambda + mu + mu1) * P).

	// H = mu * G + L + mu1 * HT + L1.
	for (int i = 0; i < H.Size(); ++i)
		*(pH++) = mu * (*(pG++)) + *(pL++) + mu1 * (*(pHT++)) + *(pL1++);

	// h = FFTInv2(H).
	GFFT.FFTInv2(H, h, true);

	// h = h * lm * P.
	float lm = 1.0f / (lambda + mu + mu1);
	for (int i = 0; i < h.Size(); ++i)
		*(ph++) *= (lm * (*(pP++)));

	// H = FFT2(h).
	GFFT.FFT2(h, H);
}

void Tracker::CalcLMHelper(const MatCF& G, const MatCF& H, MatCF& L, float mu) const {
	const ComplexF *pg = G.Data();
	const ComplexF *ph = H.Data();
	ComplexF *pl = L.Data();

	// L = L + mu * (G - H).
	for (int i = 0; i < L.Size(); ++i)
		*(pl++) += (mu * (*(pg++) - *(ph++)));
}

void Tracker::CalculateResponse(const std::vector<MatF>& ftrs, const std::vector<MatCF>& filter, const std::vector<float> &weight, MatF& response) const {
	std::vector<MatCF> Ffeatures;
	if (pcaCount > 0) {
		int ftrNum = (int)ftrs.size();
		int sampleNum = ftrs[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF pmergedFtrs(ftrs[0].Rows(), ftrs[0].Cols() * pcaCount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(ftrs);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Ffeatures, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(ftrs, Ffeatures, &arenas[ThreadIndex]);
	}
	if (Ffeatures.size() != filter.size() || filter.size() != weight.size()) {
		Critical("Tracker::CalculateResponse: feature channels not match filter channels.");
		return;
	}

	int len = (int)Ffeatures.size();
	int width = Ffeatures[0].Cols();
	int height = Ffeatures[0].Rows();

	if (useChannelWeights) {
		std::vector<MatCF> respfs(len, MatCF(&arenas[ThreadIndex]));
		std::vector<MatF> resps(len, MatF(&arenas[ThreadIndex]));
		std::vector<float> detectWeights(len, 0.0f);
		float weightSum = 0.0f;

		ParallelFor([&](int64_t k) {
			GFilter.MulSpectrums(Ffeatures[k], filter[k], respfs[k], true);
			GFFT.FFTInv2(respfs[k], resps[k]);
			float maxValue;
			Vector2i maxLoc;
			resps[k].MaxLoc(maxValue, maxLoc);

			float subValue;
			Vector2i subLoc;
			FindSubPeak(resps[k], maxLoc, subLoc, subValue);
			detectWeights[k] = std::max(1.0f - (subValue / maxValue), 0.5f) * weight[k];

//			for (int y = maxLoc.y - supressRadius; y <= maxLoc.y + supressRadius; ++y) {
//				for (int x = maxLoc.x - supressRadius; x <= maxLoc.x + supressRadius; ++x) {
//					int ny = Mod(y, height);
//					int nx = Mod(x, width);
//					*(resps[k].Data() + width * ny + nx) *= -1.0f;
//				}
//			}
//			float subMaxValue;
//			Vector2i subMaxLoc;
//			resps[k].MaxLoc(subMaxValue, subMaxLoc);
//			detectWeights[k] = std::max(1.0f - (subMaxValue / maxValue), 0.5f) * filterWeights[k];
//			for (int y = maxLoc.y - supressRadius; y <= maxLoc.y + supressRadius; ++y) {
//				for (int x = maxLoc.x - supressRadius; x <= maxLoc.x + supressRadius; ++x) {
//					int ny = Mod(y, height);
//					int nx = Mod(x, width);
//					*(resps[k].Data() + width * ny + nx) *= -1.0f;
//				}
//			}
		}, len, 2);

		response.Reshape(height, width);
		for (int i = 0; i < len; ++i) {
			weightSum += detectWeights[i];
		}
		for (int i = 0; i < len; ++i) {
			detectWeights[i] /= weightSum;
			response.CAdd(resps[i], detectWeights[i], 0.0f);
		}
	} else {
		MatCF respf(filter[0].Rows(), filter[0].Cols(), &arenas[ThreadIndex]);
		MatCF tempf(&arenas[ThreadIndex]);
		for (int i = 0; i < len; ++i) {
			GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
			respf.CAdd(tempf, 1.0f, ComplexF(0.0f, 0.0f));
		}
		GFFT.FFTInv2(respf, response);
	}

//    int len = (int)Ffeatures.size();
//    MatCF respf(filter[0].Rows(), filter[0].Cols(), &arenas[ThreadIndex]);
//    MatCF tempf(&arenas[ThreadIndex]);
//    if (useChannelWeights) {
//        for (int i = 0; i < len; ++i) {
//            GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
//            respf.CAdd(tempf, filterWeights[i], ComplexF(0.0f, 0.0f));
//        }
//    } else {
//        for (int i = 0; i < len; ++i) {
//            GFilter.MulSpectrums(Ffeatures[i], filter[i], tempf, true);
//            respf.CAdd(tempf, 1.0f, ComplexF(0.0f, 0.0f));
//        }
//    }
//    GFFT.FFTInv2(respf, response);
}

float Tracker::CalculateScore(const MatF &response, const Vector2i &pos, float maxResp) const {
	const static int offsetX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
	const static int offsetY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

	const int rows = response.Rows();
	const int cols = response.Cols();

	float distance = 0.0f;
	float score = std::numeric_limits<float>::max();
	bool peakFlag = false;
	int tx = 0, ty = 0;
	float cvalue = 0.0f, tscore = 0.0f;
	for(int r = 0; r < rows; ++r) {
		for(int c = 0; c < cols; ++c) {
			if(r == pos.y && c == pos.x) continue;
			// judge peak
			peakFlag = true;
			cvalue = response.Get(r, c);
			for(int i = 0; i < 8; ++i) {
				tx = c + offsetX[i];
				ty = r + offsetY[i];
				tx = Mod(tx, cols);
				ty = Mod(ty, rows);
				if(response.Get(ty, tx) >= cvalue) {
					peakFlag = false;
					break;
				}
			}
			if(!peakFlag) continue;

			// calculate value
			distance = std::pow(pos.x - c, 2.0f) + std::pow(pos.y - r, 2.0f);
			distance = 1.0f - std::exp(-k / 2.0f * distance);
			tscore = (maxResp - cvalue) / distance;
			if(tscore < score) score = tscore;
		}
	}

	return score;
}

void Tracker::FindSubPeak(const MatF &response, const Vector2i &peakPos, Vector2i &subPos, float &subValue) const {
	const static int offsetX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
	const static int offsetY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
	const int rows = response.Rows();
	const int cols = response.Cols();
	subValue = 0.0f;

	bool peakFlag = false;
	int tx = 0, ty = 0;
	float cvalue = 0.0f;
	for(int r = 0; r < rows; ++r) {
		for(int c = 0; c < cols; ++c) {
			if(r == peakPos.y && c == peakPos.x) continue;
			// judge peak
			peakFlag = true;
			cvalue = response.Get(r, c);
			for(int i = 0; i < 8; ++i) {
				tx = c + offsetX[i];
				ty = r + offsetY[i];
				tx = Mod(tx, cols);
				ty = Mod(ty, rows);
				if(response.Get(ty, tx) >= cvalue) {
					peakFlag = false;
					break;
				}
			}
			if(!peakFlag) continue;

			// calculate value
			if(cvalue > subValue) {
				subValue = cvalue;
				subPos.x = c;
				subPos.y = r;
			}
		}
	}
}

void Tracker::ResetArenas(bool background) const {
	if (!arenas) return;
	int len = MaxThreadIndex() - 1;
	if (background) {
		arenas[len].Reset();
	} else {
		for (int i = 0; i < len; ++i)
			arenas[i].Reset();
	}
}

//
// Tracker Main Interface
//

void Tracker::Initialize(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask, float orgArea) {
	if (initialized) return;
	initialized = true;
	if (!arenas)
		arenas = new MemoryArena[MaxThreadIndex()];

	// Input check.
	if (feats.size() <= 0 || feats[0].Size() <= 0 || filterMask.Size() <= 0) {
		Critical("Tracker::Initialize: cannot initialize with empty features data or filterMask.");
		return;
	}
	if (yf.Rows() != filterMask.Rows() || yf.Cols() != filterMask.Cols()) {
		Critical("Tracker::Initialize: ideal response and filter mask not match.");
		return;
	}
	if(orgArea <= 0.0f) {
		Critical("Tracker::Initialize: invalid target area.");
		return;
	}
	targetArea = orgArea;
	k = 8.0f / targetArea;

	// Initialize filter.
	std::vector<MatCF> Fftrs;
	if (pcaCount > (int)feats.size()) {
		Warning("Tracker::Initialize: PCA count is greater than original feature count. Disable PCA.");
		pcaCount = 0;
	}
	if (pcaCount > 0) {
		// PCA features.
		int ftrNum = (int)feats.size();
		int sampleNum = feats[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF centerdFtrs(sampleNum, ftrNum, &arenas[ThreadIndex]);
		MatF cov(ftrNum, ftrNum, &arenas[ThreadIndex]);
		MatF pmergedFtrs(feats[0].Rows(), feats[0].Cols() * pcaCount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(feats);
		project.Reshape(ftrNum, pcaCount, false);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egCenteredM(centerdFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egCovM(cov.Data(), ftrNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
		egCenteredM = egDataM.rowwise() - egDataM.colwise().mean();
		egCovM.noalias() = egCenteredM.transpose() * egCenteredM;
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(egCovM);
		if (eig.info() != Eigen::Success) {
			Critical("Cannot solve eigen values for PCA process.");
			return;
		}
		egProjM = eig.eigenvectors().rightCols(pcaCount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(feats, Fftrs, &arenas[ThreadIndex]);
	}
	CreateFilters(Fftrs, yf, filterMask, filters[0]);

	// Initialize channel weights.
	if (useChannelWeights) {
		int len = (int)filters[0].size();
		filterWeights[0].resize(len, 0.0f);
		ParallelFor([&](int64_t index) {
			MatCF currentRespf(&arenas[ThreadIndex]);
			MatF currentResp(&arenas[ThreadIndex]);
			float maxValue = 0.0f;
			Vector2i maxLoc;
			GFilter.MulSpectrums(Fftrs[index], filters[0][index], currentRespf, true);
			GFFT.FFTInv2(currentRespf, currentResp, true);
			currentResp.MaxLoc(maxValue, maxLoc);
			filterWeights[0][index] = maxValue;
		}, len, 1);
		float sumWeights = 0.0f;
		for (int i = 0; i < len; ++i)
			sumWeights += filterWeights[0][i];
		for (int i = 0; i < len; ++i)
			filterWeights[0][i] /= sumWeights;
	}

	int total = (int)filters.size();
	for(int i = 1; i < total; ++i) {
		filters[i] = filters[0];
		filterWeights[i] = filterWeights[0];
	}

	ResetArenas(false);
	ResetArenas(true);
}

void Tracker::GetPosition(const std::vector<MatF> &feats, float currentScale, int cellSize, float rescaleRatio,
						  const Vector2f &currentPos, const Vector2i &moveSize, Vector2f &newPos, float &score) const {
	if (!initialized) {
		Error("Tracker::GetPosition: tracker is not initialized.");
		return;
	}

	int len = (int)filters.size();
	MatF response(&arenas[ThreadIndex]);
	std::vector<MatF> tempResponses(len, MatF(&arenas[ThreadIndex]));
	std::vector<float> scores(len);
	float maxValue;
	Vector2i maxLoc;
	float scoreSum = 0.0f;
	for(int i = 0; i < len; ++i) {
		CalculateResponse(feats, filters[i], filterWeights[i], tempResponses[i]);
		tempResponses[i].MaxLoc(maxValue, maxLoc);
		scores[i] = CalculateScore(tempResponses[i], maxLoc, maxValue);
		scoreSum += scores[i];
	}
	response.Reshape(tempResponses[0].Rows(), tempResponses[0].Cols());
	for (int i = 0; i < len; ++i) {
		response.CAdd(tempResponses[i], scores[i] / scoreSum, 0.0f);
	}
	response.MaxLoc(maxValue, maxLoc);
	score = CalculateScore(response, maxLoc, maxValue);

	// Take into account also subpixel accuracy.
	float x = maxLoc.x + GFilter.SubpixelPeak(response, maxLoc, SubpixelDirection::Horizontal);
	float y = maxLoc.y + GFilter.SubpixelPeak(response, maxLoc, SubpixelDirection::Vertical);
	if (x > (response.Cols() - 1) / 2.0f) x -= response.Cols();
	if (y > (response.Rows() - 1) / 2.0f) y -= response.Rows();

	// Calculate new position.
	float scale = currentScale * cellSize / rescaleRatio;
	newPos = currentPos + Vector2f(scale * x, scale * y);

	// Fit to bound.
	newPos.x = Clamp(newPos.x, 0.0f, moveSize.x - 1.0f);
	newPos.y = Clamp(newPos.y, 0.0f, moveSize.y - 1.0f);

	ResetArenas(false);
}

void Tracker::Update(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask) {
	if (!initialized) {
		Error("Tracker::Update: tracker is not initialized.");
		return;
	}

	// Input check.
	if (feats.size() <= 0 || feats[0].Size() <= 0 || filterMask.Size() <= 0) {
		Critical("Tracker::Initialize: cannot initialize with empty features data or filterMask.");
		return;
	}
	if (yf.Rows() != filterMask.Rows() || yf.Cols() != filterMask.Cols()) {
		Critical("Tracker::Initialize: ideal response and filter mask not match.");
		return;
	}

	std::vector<MatCF> Fftrs;
	if (pcaCount > 0) {
		int ftrNum = (int)feats.size();
		int sampleNum = feats[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF pmergedFtrs(feats[0].Rows(), feats[0].Cols() * pcaCount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(feats);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(feats, Fftrs, &arenas[ThreadIndex]);
	}
	std::vector<MatCF> newFilters;
	CreateFilters(Fftrs, yf, filterMask, newFilters, &arenas[ThreadIndex]);

	// Calculate per channel weights.
	int len = (int)newFilters.size();
	int num = (int)filters.size();
	if (useChannelWeights) {
		std::vector<float> newFilterWeights(newFilters.size(), 0.0f);
		ParallelFor([&](int64_t index) {
			MatCF currentRespf(&arenas[ThreadIndex]);
			MatF currentResp(&arenas[ThreadIndex]);
			float maxValue = 0.0f;
			Vector2i maxLoc;
			GFilter.MulSpectrums(Fftrs[index], newFilters[index], currentRespf, true);
			GFFT.FFTInv2(currentRespf, currentResp, true);
			currentResp.MaxLoc(maxValue, maxLoc);
			newFilterWeights[index] = maxValue;
		}, len, 1);
		float sumWeights = 0.0f;
		for (int i = 0; i < len; ++i)
			sumWeights += newFilterWeights[i];

		// Update filter weights with new values.
		float sumUpdate;
		for(int k = 0; k < num; ++k) {
			sumUpdate = 0.0f;
			for (int i = 0; i < len; ++i) {
				filterWeights[k][i] = Lerp(weightsLearnRates[k], filterWeights[k][i], newFilterWeights[i] / sumWeights);
				sumUpdate += filterWeights[k][i];
			}
			// Normalize weights.
			for (int i = 0; i < len; ++i)
				filterWeights[k][i] /= sumUpdate;
		}
	}

	// Update filters.
	ParallelFor([&](int64_t i) {
		for(int k = 0; k < num; ++k) {
			filters[k][i].Lerp(newFilters[i], filterLearnRates[k]);
		}
	}, len, 2);

	ResetArenas(false);
}

void Tracker::StartBackgroundUpdate() {
	if (backgroundUpdateFlag) return;
	bk_filters = filters;
	bk_filterWeights = filterWeights;
	backgroundUpdateFlag = true;
}

void Tracker::FetchUpdateResult() {
	if (!backgroundUpdateFlag) return;
	filters = bk_filters;
	filterWeights = bk_filterWeights;
	backgroundUpdateFlag = false;
}

void Tracker::BackgroundUpdate(const std::vector<MatF> &feats, const MatCF &yf, const MatF &filterMask) {
	// Extract features.
	std::vector<MatCF> Fftrs;
	if (pcaCount > 0) {
		int ftrNum = (int)feats.size();
		int sampleNum = feats[0].Size();
		MatF mergedFtrs(&arenas[ThreadIndex]);
		MatF pmergedFtrs(feats[0].Rows(), feats[0].Cols() * pcaCount, &arenas[ThreadIndex]);
		mergedFtrs.Merge(feats);
		Eigen::Map<Eigen::MatrixXf> egDataM(mergedFtrs.Data(), sampleNum, ftrNum);
		Eigen::Map<Eigen::MatrixXf> egProjM(project.Data(), ftrNum, pcaCount);
		Eigen::Map<Eigen::MatrixXf> egProjDataM(pmergedFtrs.Data(), sampleNum, pcaCount);
		egProjDataM.noalias() = egDataM * egProjM;
		std::vector<MatF> pftrs;
		pmergedFtrs.Split(pftrs, pcaCount, &arenas[ThreadIndex]);
		GFilter.FFT2Collection(pftrs, Fftrs, &arenas[ThreadIndex]);
	} else {
		GFilter.FFT2Collection(feats, Fftrs, &arenas[ThreadIndex]);
	}
	std::vector<MatCF> newFilters;
	CreateFilters(Fftrs, yf, filterMask, newFilters, &arenas[ThreadIndex]);

	// Calculate per channel weights.
	int len = (int)newFilters.size();
	int num = (int)filters.size();
	if (useChannelWeights) {
		std::vector<float> newFilterWeights(newFilters.size(), 0.0f);
		ParallelFor([&](int64_t index) {
			MatCF currentRespf(&arenas[ThreadIndex]);
			MatF currentResp(&arenas[ThreadIndex]);
			float maxValue = 0.0f;
			Vector2i maxLoc;
			GFilter.MulSpectrums(Fftrs[index], newFilters[index], currentRespf, true);
			GFFT.FFTInv2(currentRespf, currentResp, true);
			currentResp.MaxLoc(maxValue, maxLoc);
			newFilterWeights[index] = maxValue;
		}, len, 1);
		float sumWeights = 0.0f;
		for (int i = 0; i < len; ++i)
			sumWeights += newFilterWeights[i];

		// Update filter weights with new values.
		float sumUpdate;
		for (int k = 0; k < num; ++k) {
			sumUpdate = 0.0f;
			for (int i = 0; i < len; ++i) {
				bk_filterWeights[k][i] = Lerp(weightsLearnRates[k], bk_filterWeights[k][i], newFilterWeights[i] / sumWeights);
				sumUpdate += bk_filterWeights[k][i];
			}
			// Normalize weights.
			for (int i = 0; i < len; ++i)
				bk_filterWeights[k][i] /= sumUpdate;
		}
	}

	// Update filters.
	ParallelFor([&](int64_t i) {
		for (int k = 0; k < num; ++k) {
			bk_filters[k][i].Lerp(newFilters[i], filterLearnRates[k]);
		}
	}, len, 2);

	ResetArenas(true);
}

}	// namespace CSRT