#include "../Include/FFTCC.h"
#include "../Include/MemManager.h"

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void fftccGuess::init(
	const int threadNum,
	float *fVolR, float *fVolT,
	int iOriginVolWidth, int iOriginVolHeight, int iOriginVolDepth) {

	thread_num = threadNum;
	m_fVolR = fVolR;
	m_fVolT = fVolT;
	m_iOriginVolWidth = iOriginVolWidth;
	m_iOriginVolHeight = iOriginVolHeight;
	m_iOriginVolDepth = iOriginVolDepth;

	if (m_iSubsetX <= 0 || m_iSubsetY <= 0 || m_iSubsetZ <= 0) {
		cerr << "Invalid subset radii in fftccGuess::init()" << endl;
		throw "Invalid Subset Radii";
	}

	const int iFFTSubW = m_iSubsetX * 2;
	const int iFFTSubH = m_iSubsetY * 2;
	const int iFFTSubD = m_iSubsetZ * 2;
	//int iNumberofPOI1Batch = m_iNumberX*m_iNumberY*m_iNumberZ;

	//CMemManager<float>::hCreatePtr(m_fSubset1, thread_num*iFFTSubD*iFFTSubH*iFFTSubW);
	//CMemManager<float>::hCreatePtr(m_fSubset2, thread_num*iFFTSubD*iFFTSubH*iFFTSubW);
	//CMemManager<float>::hCreatePtr(m_fSubsetC, thread_num*iFFTSubD*iFFTSubH*iFFTSubW);

	m_FreqDom1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*thread_num*iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1));
	m_FreqDom2 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*thread_num*iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1));
	m_FreqDomfg = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*thread_num*iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1));

	m_fftwPlan1 = new fftwf_plan[thread_num];
	m_fftwPlan2 = new fftwf_plan[thread_num];
	m_rfftwPlan = new fftwf_plan[thread_num];

	for (int i = 0; i < thread_num; i++)
	{
		m_fftwPlan1[i] = fftwf_plan_dft_r2c_3d(iFFTSubW, iFFTSubH, iFFTSubD, &m_fSubset1[i*iFFTSubD*iFFTSubH*iFFTSubW], &m_FreqDom1[i*iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1)], FFTW_ESTIMATE);
		m_fftwPlan2[i] = fftwf_plan_dft_r2c_3d(iFFTSubW, iFFTSubH, iFFTSubD, &m_fSubset2[i*iFFTSubD*iFFTSubH*iFFTSubW], &m_FreqDom2[i*iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1)], FFTW_ESTIMATE);
		m_rfftwPlan[i] = fftwf_plan_dft_c2r_3d(iFFTSubW, iFFTSubH, iFFTSubD, &m_FreqDomfg[i*iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1)], &m_fSubsetC[i*iFFTSubD*iFFTSubH*iFFTSubW], FFTW_ESTIMATE);
	}
}

void fftccGuess::free() {
	if (m_fSubset1 != nullptr)
		//CMemManager<float>::hDestroyPtr(m_fSubset1);
	if (m_fSubset2 != nullptr)
		//CMemManager<float>::hDestroyPtr(m_fSubset2);
	if (m_fSubsetC != nullptr)
		//CMemManager<float>::hDestroyPtr(m_fSubsetC);

	if (m_fftwPlan1 != nullptr) {
		for (int i = 0; i < thread_num; i++)
		{
			fftwf_destroy_plan(m_fftwPlan1[i]);
			fftwf_destroy_plan(m_fftwPlan2[i]);
			fftwf_destroy_plan(m_rfftwPlan[i]);
		}
	}

	if (m_fftwPlan1 != nullptr)
		delete m_fftwPlan1;
	m_fftwPlan1 = nullptr;

	if (m_fftwPlan2 != nullptr)
		delete m_fftwPlan2;
	m_fftwPlan2 = nullptr;

	if (m_rfftwPlan != nullptr)
		delete m_rfftwPlan;
	m_rfftwPlan = nullptr;

	fftwf_free(m_FreqDom1);
	fftwf_free(m_FreqDom2);
	fftwf_free(m_FreqDomfg);
}

void fftccGuess::compute(CPOI &POI_) {

	const int iID_ = omp_get_thread_num();

	//StopWatchWin fftccWatch;
	POI_.SetProcessed(1);

	int iFFTSubW = m_iSubsetX * 2;
	int iFFTSubH = m_iSubsetY * 2;
	int iFFTSubD = m_iSubsetZ * 2;
	int iFFTSize = iFFTSubW * iFFTSubH * iFFTSubD;
	int iFFTFreqSize = iFFTSubW * iFFTSubH*(iFFTSubD / 2 + 1);
	int iSubsetX = m_iSubsetX;
	int iSubsetY = m_iSubsetY;
	int iSubsetZ = m_iSubsetZ;

	int iCorrPeak, m_iU, m_iV, m_iW;
	float m_fSubAveR, m_fSubAveT, m_fSubNorR, m_fSubNorT;

	m_fSubAveR = 0;	// R_m
	m_fSubAveT = 0;	// T_m

					// Start timer for FFT-CC algorithm
					//fftccWatch.start();

					// Feed the grey intensity values into subvolumes
	float ref_voxel, tar_voxel;
	for (int l = 0; l < iFFTSubD; l++)
	{
		for (int m = 0; m < iFFTSubH; m++)
		{
			for (int n = 0; n < iFFTSubW; n++)
			{
				ref_voxel = m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth,
					POI_.G_X() - iSubsetX + n,
					POI_.G_Y() - iSubsetY + m,
					POI_.G_Z() - iSubsetZ + l)];
				m_fSubset1[iID_*iFFTSize + ELT(iFFTSubH, iFFTSubW, n, m, l)] = ref_voxel;
				m_fSubAveR += ref_voxel;

				tar_voxel = m_fVolT[ELT(m_iOriginVolHeight, m_iOriginVolWidth,
					POI_.G_X() - iSubsetX + n,
					POI_.G_Y() - iSubsetY + m,
					POI_.G_Z() - iSubsetZ + l)];
				m_fSubset2[iID_*iFFTSize + ELT(iFFTSubH, iFFTSubW, n, m, l)] = tar_voxel;
				m_fSubAveT += tar_voxel;
			}
		}
	}
	m_fSubAveR = m_fSubAveR / float(iFFTSize);
	m_fSubAveT = m_fSubAveT / float(iFFTSize);

	m_fSubNorR = 0;		// sqrt (Sigma(R_i - R_m)^2)
	m_fSubNorT = 0;		// sqrt (Sigma(T_i - T_m)^2)

	for (int l = 0; l < iFFTSubD; l++)
	{
		for (int m = 0; m < iFFTSubH; m++)
		{
			for (int n = 0; n < iFFTSubW; n++)
			{
				m_fSubset1[iID_*iFFTSize + ELT(iFFTSubH, iFFTSubW, n, m, l)] -= m_fSubAveR;
				m_fSubset2[iID_*iFFTSize + ELT(iFFTSubH, iFFTSubW, n, m, l)] -= m_fSubAveT;
				m_fSubNorR += pow((m_fSubset1[iID_*iFFTSize + ELT(iFFTSubH, iFFTSubW, n, m, l)]), 2);
				m_fSubNorT += pow((m_fSubset2[iID_*iFFTSize + ELT(iFFTSubH, iFFTSubW, n, m, l)]), 2);
			}
		}
	}

	// Terminate the processing if one of the two subvolumes is full of zero intensity
	if (m_fSubNorR == 0 || m_fSubNorT == 0)
	{
		POI_.SetDarkSubset(1); //set flag
		return;
	}

	//Question here?
	// FFT-CC algorithm accelerated by FFTW
	fftwf_execute(m_fftwPlan1[iID_]);
	fftwf_execute(m_fftwPlan2[iID_]);
	for (int p = 0; p < iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1); p++)
	{
		m_FreqDomfg[iID_*iFFTFreqSize + p][0] = (m_FreqDom1[iID_*iFFTFreqSize + p][0] * m_FreqDom2[iID_*iFFTFreqSize + p][0])
			+ (m_FreqDom1[iID_*iFFTFreqSize + p][1] * m_FreqDom2[iID_*iFFTFreqSize + p][1]);
		m_FreqDomfg[iID_*iFFTFreqSize + p][1] = (m_FreqDom1[iID_*iFFTFreqSize + p][0] * m_FreqDom2[iID_*iFFTFreqSize + p][1])
			- (m_FreqDom1[iID_*iFFTFreqSize + p][1] * m_FreqDom2[iID_*iFFTFreqSize + p][0]);
	}
	fftwf_execute(m_rfftwPlan[iID_]);

	float fTempZNCC = -2;	// Maximum C
	iCorrPeak = 0;			// Location of maximum C
							// Search for maximum C, then normalize C
	for (int k = 0; k < iFFTSubW*iFFTSubH*iFFTSubD; k++)
	{
		if (fTempZNCC < m_fSubsetC[iID_*iFFTSize + k])
		{
			fTempZNCC = m_fSubsetC[iID_*iFFTSize + k];
			iCorrPeak = k;
		}
	}

	fTempZNCC /= sqrt(m_fSubNorR * m_fSubNorT)*float(iFFTSize); //parameter for normalization

																// calculate the loacation of maximum C
	m_iU = iCorrPeak % iFFTSubW;
	m_iV = iCorrPeak / iFFTSubW % iFFTSubH;
	m_iW = iCorrPeak / iFFTSubW / iFFTSubH;
	// Shift the C peak to the right quadrant 
	if (m_iU > m_iSubsetX)
		m_iU -= iFFTSubW;
	if (m_iV > m_iSubsetY)
		m_iV -= iFFTSubH;
	if (m_iW > m_iSubsetZ)
		m_iW -= iFFTSubD;
	// P0[u,ux,uy,uz,v,vx,vy,vz,w,wx,wy,wz]
	vector<float>tempP0(12, 0);
	tempP0[0] = float(m_iU);
	tempP0[4] = float(m_iV);
	tempP0[8] = float(m_iW);

	POI_.SetZNCC(fTempZNCC);
	POI_.SetP0(tempP0);
	POI_.strategy = FFTCC;
}