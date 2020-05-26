#ifndef __PIDVC_MULDVC_H__
#define __PIDVC_MULDVC_H__

#include <vector>
#include <array>
#include <string>

#include <3DSIFT/Include/cSIFT3D.h>

#include "POI.h"
#include <fftw3.h>
#include <kdtree.h>
#include "PriorityQueue.h"
#include "FFTCC.h"
#include "FitFormula.h"
#include "ICGN.h"

#define RADIUS_FACTOR 1.2f
#define ZNCC_THRES 0.8f
#define MAX_KP_ICGN_ITERA 8
#define ZNCC_THRES_STRATEGY2 0.8f
#define ZNCC_THRES_STRATEGY3 0.8f
#define SIGMA_THRES 4.0f

enum INTERPOLATION_METHOD {
	TRICUBIC,
	BSPLINE
};

enum GuessMethod {
	PiSIFT = 1,
	IOPreset = 2,
	FFTCCGuess = 3,
};

float dist_square_Cvec(const CPUSIFT::Cvec &c1, const CPUSIFT::Cvec &c2);

//-! Structure to store time consumed 
struct TimerPara
{
	double dPreparationTime = 0.0;
	double dPrecomputationTime = 0.0;

	//guess
	double dBuildKDTreeTime = 0.0;
	double dLocalAffineTime = 0.0;
	double dFFTCcTime = 0.0;

	//iterative
	double dICGNTime = 0.0;
	
	//strategy 3
	double dTransferStrategyTime = 0.0;
	
	double dConsumedTime = 0.0;
	double dFinalizationTime = 0.0;
};

//-! CPaDVC class definition
class CPaDVC
{
public:

	std::vector<CPOI > m_POI_global;
	//-! Save the matching point from SIFT3D
	std::vector<CPUSIFT::Cvec> Ref_point;
	std::vector<CPUSIFT::Cvec> Tar_point;
	//Volume image R,T 
	float *m_fVolR = nullptr;
	float *m_fVolT = nullptr;
	int m_iOriginVolWidth;
	int m_iOriginVolHeight;
	int m_iOriginVolDepth;

	INTERPOLATION_METHOD m_eInterpolation = BSPLINE;
	TimerPara m_Timer;				// Timer used to calculate each algorithm

	CPaDVC(const std::string chFile1, const std::string chFile2, const int_3d i3dSubVol, int thread_num);

	~CPaDVC();

	void ReadVol(const std::string chFile1, const std::string chFile2);

	void KD_NeighborPOI_assign_6NN();

	void SetPiSIFTInitParam(const int minNeighNum, const float errorEpsilon, const int maxIter);

	void SetICGNParam(const float deltaP, const int maxIter);

	//algorithm
	void PiDVCAlgorithm_mul_global_STRATEGY();

	void PiDVCAlgorithm_mul_global_FFTCC();

	void PiDVCAlgorithm_mul_global_ICGN_Only();

	void SaveResult2Text_global_d(
		const std::string cstrOutputPath_,
		SIFT_PROCESS SIFT_TIME, GuessMethod initMethod,
		const std::string  c_ref_ = std::string(""), const std::string  c_tar_ = std::string(""),
		const std::string  f_ref_ = std::string(""), const std::string  f_tar_ = std::string(""));



private:
	fftccGuess fftccGuessCompute;
	siftGuess siftGuessCompute;
	icgnRegistration icgnCompute;
	//mulFitFormula ransacCompute;

	void Destroy();

	// -! Prepreation & finalization
	void Precomputation_Prepare_global();
	void Precomputation_Finalize();

	// -! Precomputation
	void PrecomputeGradients_mul_global();
	void TransSplineArray();

	// -! funs in Strategy3
	int Collect_neighborPOI(CPOI &POI_, int *Neighbour_id);
	int Collect_neighborBestPOI(CPOI &POI_, int *Neighbour_id);
	void BatchICGN(std::vector<int> toBe, int tnum);
	void Strategy3_simple_transfer();


private:			
	//-! ROI & its gradient of Voxel R, Bspline coeffi for Voxel T
	float ***m_fR = nullptr;
	float ***m_fRx = nullptr;
	float ***m_fRy = nullptr;
	float ***m_fRz = nullptr;
	float ***m_fTGBspline = nullptr;

	//kdtree for POI to find neighbor POI, when using ioPreset
	kdtree *kd_POI = nullptr;
	int *kd_POI_idx = nullptr;

	//Internal Parameters for DVC
	bool m_bIsExecuted;
	int m_iOriginROIWidth;			//=m_iOriginVolWidth-2
	int m_iOriginROIHeight;			//=m_iOriginVolHeight-2
	int m_iOriginROIDepth;			//=m_iOriginVolDepth-2
	
	int m_iWholeTotalIterations = 0;
	
	//-! subset size 
	int m_iSubsetX;
	int m_iSubsetY;
	int m_iSubsetZ;
	
	int thread_num;

public:
	auto originWidth() { return m_iOriginVolWidth; }
	auto originHeight() { return m_iOriginVolHeight; }
	auto originDepth() { return m_iOriginVolDepth; }
	
};

class CCUPaDVCFactory
{
public:
	//	CPaDVC(const std::string chFile1, const std::string chFile2, const int_3d i3dSubVol, int thread_num);

	static CPaDVC* CreateCUPaDVC(const std::string chFile1, const std::string chFile2, const int_3d i3dSubVol, int thread_num)
	{
		return (new CPaDVC(chFile1, chFile2, i3dSubVol, thread_num));
	}
};

#endif // ! MULDVC_H
