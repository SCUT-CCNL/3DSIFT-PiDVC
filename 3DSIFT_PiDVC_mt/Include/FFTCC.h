#ifndef __PIDVC_FFTCC_H__
#define __PIDVC_FFTCC_H__

#include "compute.h"
#include "POI.h"

#include "fftw3.h"

class fftccGuess : public computePOI {

	//-!Volume
	float *m_fVolR = nullptr;
	float *m_fVolT = nullptr;
	int m_iOriginVolWidth = 0;
	int m_iOriginVolHeight = 0;
	int m_iOriginVolDepth = 0;

	//-!FFT-CC parameters
	float *m_fSubset1 = nullptr;				// POI_1batch* [iFFTSubW * iFFTSubH * iFFTSubD]
	float *m_fSubset2 = nullptr;				// POI_1batch* [iFFTSubW * iFFTSubH * iFFTSubD]
	float *m_fSubsetC = nullptr;				// POI_1batch* [iFFTSubW * iFFTSubH * iFFTSubD]
	fftwf_plan *m_fftwPlan1 = nullptr;
	fftwf_plan *m_fftwPlan2 = nullptr;
	fftwf_plan *m_rfftwPlan = nullptr;
	fftwf_complex	*m_FreqDom1 = nullptr;	// POI_1batch* [iFFTSubW * iFFTSubH * (iFFTSubD/2 + 1)]
	fftwf_complex	*m_FreqDom2 = nullptr;	// POI_1batch* [iFFTSubW * iFFTSubH * (iFFTSubD/2 + 1)]
	fftwf_complex	*m_FreqDomfg = nullptr;	// POI_1batch* [iFFTSubW * iFFTSubH * (iFFTSubD/2 + 1)]

public:
	fftccGuess() {};
	~fftccGuess() {};

	void init(const int threadNum,
		float *fVolR, float *fVolT,
		int iOriginVolWidth, int iOriginVolHeight, int iOriginVolDepth);

	void preCompute() {};
	void free();
	void compute(CPOI &POI_);
}

#endif
