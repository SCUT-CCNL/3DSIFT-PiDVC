#ifndef __PIDVC_ICGN_REGISTRATION_H__
#define __PIDVC_ICGN_REGISTRATION_H__

#include "compute.h"
#include "POI.h"
#include <vector>

class icgnRegistration : public computePOI {
	
	//volume data
	float *m_fVolR = nullptr;
	float *m_fVolT = nullptr;
	int m_iOriginVolWidth = 0;
	int m_iOriginVolHeight = 0;
	int m_iOriginVolDepth = 0;

	//Gradients
	float ***m_fRx = nullptr;
	float ***m_fRy = nullptr;
	float ***m_fRz = nullptr;
	//Bspline coefficients
	float ***m_fTGBspline = nullptr;

	//temeporay data
	float ****m_fSubsetR = nullptr;							// ICGN Subset of Resourse 
	float ****m_fSubsetT = nullptr;							// ICGN Subset of Target
	float *****m_fRDescent = nullptr;							// ICGN deltaR * (dW/dP);

	//parameters
	float m_dDeltaP;
	int m_iMaxIterationNum;

	//compute func
	int InverseHessian_GaussianJordan(
		std::vector<std::vector<float>>&m_fInvHessian, 
		std::vector<std::vector<float>>&m_fHessian);
public:
	icgnRegistration() {};
	~icgnRegistration() {};

	void init(const int threadNum,
		float *fVolR, float *fVolT,
		int iOriginVolWidth, int iOriginVolHeight, int iOriginVolDepth,
		float ***fRx, float ***fRy, float ***fRz, 
		float ***fTGBspline);

	void setParameters(const float deltaP, const int maxIter) {
		m_dDeltaP = deltaP;
		m_iMaxIterationNum = maxIter;
	};

	void preCompute() {};
	void free();
	void compute(CPOI &POI_);

	float getDeltaP() { return m_dDeltaP; };
	int getMaxIter() { return m_iMaxIterationNum; };

};

#endif