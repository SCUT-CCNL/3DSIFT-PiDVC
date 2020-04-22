#include "mulDVC.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <deque>
#include <random>
#include <thread>
#include <ctime>
#include <map>

#include <omp.h>
#include <Eigen\core>
#include <Eigen\dense>
#include <Eigen\LU>
#include <mutex>

#include "fftw3.h"
#include "MemManager.h"
#include "matrixIO3D.h"
#include "util.h"
#include "FitFormula.h"
#include "Spline.h"

using namespace std;
using namespace CPUSIFT;

//#define _JR_DEBUG

#ifdef _DEBUG
	int debugP[3] = { 170,160, 8 };
#endif

/*
	tools func used in calculation
*/
inline void CvecSubtract(Cvec &c1, const Cvec &c2){
	c1.x -= c2.x;
	c1.y -= c2.y;
	c1.z -= c2.z;
}

void TransferLocalCoor(vector<Cvec> &vc, const Cvec origin){
	for(auto &ci : vc)
		CvecSubtract(ci, origin);
}

float maxDistanceSquareTo(const vector<Cvec> &vc, const Cvec &target){
	float maxDistSq = 0.0;
	for(auto ci: vc){
		float dist_sq = dist_square_Cvec(ci, target);
		if(dist_sq>maxDistSq)
			maxDistSq = dist_sq;
	}
	return maxDistSq;
}

inline float dist_Cvec(const Cvec &c1, const Cvec &c2){
	return sqrtf(dist_square_Cvec(c1, c2));
}

inline float dist_square_Cvec(const Cvec &c1, const Cvec &c2) {
	float dx = c1.x - c2.x;
	float dy = c1.y - c2.y;
	float dz = c1.z - c2.z;
	return dx*dx + dy*dy + dz*dz;
}

vector<float> Calculate_P0_full(CPOI &ControlPoint, CPOI &POI_) {

	auto P = ControlPoint.P();

	float deltaX = POI_.G_X() - ControlPoint.G_X();
	float deltaY = POI_.G_Y() - ControlPoint.G_Y();
	float deltaZ = POI_.G_Z() - ControlPoint.G_Z();

	vector<float> temp_P0 = P;

	temp_P0[0] = P[0] + deltaX*P[1] + deltaY*P[2] + deltaZ*P[3];
	temp_P0[4] = P[4] + deltaX*P[5] + deltaY*P[6] + deltaZ*P[7];
	temp_P0[8] = P[8] + deltaX*P[9] + deltaY*P[10] + deltaZ*P[11];

	return temp_P0;
}

void time_info2(const char* info)
{

	static double complete_start = omp_get_wtime();
	static double total_time = -1.0f;
	static double init = -1.0f;

	//checkerror(info);
	double now = omp_get_wtime();
	if (init < 0)
	{
		init = now;
		total_time = now;
	}


	//image finish
	if (info[0] == '@')
	{
		cout << "\ttotal time:" << (now - total_time) << "s  ----" << info + 1 << endl;
		init = now;
		total_time = now;
		return;
	}


	cout << "\t\ttime:" << 1000 * (now - init) << "ms  ----" << info << endl;
	init = now;

}

/*
	definitions of funs in class CPaDVC
*/

CPaDVC::CPaDVC(const std::string chFile1, const std::string chFile2, const int_3d i3dSubVol, int thread_num) : 
	//initialization 
	m_iSubsetX(i3dSubVol.x), m_iSubsetY(i3dSubVol.y), m_iSubsetZ(i3dSubVol.z),
	m_bIsExecuted(false), m_iWholeTotalIterations(0), thread_num(thread_num){

	ReadVol(chFile1, chFile2);

	m_iExpandRatio = 1.2;
	m_iSubsetXExpand = static_cast<int>(static_cast<float>(m_iSubsetX) * (m_iExpandRatio + FLT_EPSILON));
	m_iSubsetYExpand = static_cast<int>(static_cast<float>(m_iSubsetY) * (m_iExpandRatio + FLT_EPSILON));
	m_iSubsetZExpand = static_cast<int>(static_cast<float>(m_iSubsetZ) * (m_iExpandRatio + FLT_EPSILON));

	omp_set_num_threads(thread_num);

	//!= 6. Output the configuration parameters
	cout << "Configuration Parameters: " << endl;
	cout << "Whole ROIWidth: " << m_iOriginROIWidth << endl;
	cout << "Whole ROIHeight: " << m_iOriginROIHeight << endl;
	cout << "Whole ROIDepth: " << m_iOriginROIDepth << endl;
}

CPaDVC::~CPaDVC()
{
	//-! Destroy the "float" parameters
	if (m_bIsExecuted)
	{
		Precomputation_Finalize();
		FFTCC_Finalize();
		Affine_Finalize_kdtree();
		ICGN_Finalize();
		Destroy();
	}
	else
	{
		Destroy();
	}
}

void CPaDVC::ReadVol(const string chFile1, const string chFile2) {

	//-! 1. Load and compare two voxels
	int m, n, p;
	if (ReadMatrixFromDisk(chFile1.c_str(), &m_iOriginVolWidth, &m_iOriginVolHeight, &m_iOriginVolDepth, &m_fVolR) != 0)
	{
		cout << "Error in loading matrix from " << chFile1 << endl;
		throw 1;
	}
	if (ReadMatrixFromDisk(chFile2.c_str(), &m, &n, &p, &m_fVolT) != 0)
	{
		cout << "Error in loading matrix from " << chFile2 << endl;
		throw 1;
	}
	if (m != m_iOriginVolWidth || n != m_iOriginVolHeight || p != m_iOriginVolDepth)
	{
		cout << "Error! The dimension of two matrix did not match!" << endl;
		throw 2;
	}

	//AMENDED:
	//!= 2. Initialize the original parameters
	m_iOriginROIWidth = m_iOriginVolWidth - 2;
	m_iOriginROIHeight = m_iOriginVolHeight - 2;
	m_iOriginROIDepth = m_iOriginVolDepth - 2;
}

void CPaDVC::Destroy()
{
	CMemManager<float>::hDestroyPtr(m_fVolR);
	CMemManager<float>::hDestroyPtr(m_fVolT);
}

void CPaDVC::KD_NeighborPOI_assign_6NN() {
	vector<Cvec> v_POIcoor;
	for (auto &poi_ : m_POI_global) {
		Cvec tmp(poi_.G_X(), poi_.G_Y(), poi_.G_Z());
		v_POIcoor.push_back(tmp);
	}

	KD_Build(kd_POI, kd_POI_idx, v_POIcoor);

#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int id_ = 0; id_ < m_POI_global.size(); ++id_) {
		auto &poi_ = m_POI_global[id_];
		if (poi_.GetProcessed() != 1) {
			vector<Cvec> neighbor;
			vector<int> neighborIdx;
			Cvec query(poi_.G_X(), poi_.G_Y(), poi_.G_Z());
			KD_KNNSerach(kd_POI, neighbor, neighborIdx, query, 7);
			int j = 0;
			for (int i = 0; i < MIN(neighborIdx.size(), 7); ++i) {
				if (neighborIdx[i] != id_) {
					poi_.neighbor[j] = neighborIdx[i];
					++j;
				}
			}
		}
	}
#pragma omp barrier

	KD_Destroy(kd_POI, kd_POI_idx);
}

void CPaDVC::SetPiSIFTInitParam(const int minNeighNum, const float errorEpsilon, const int maxIter) {
	ransacCompute.setParam(errorEpsilon, maxIter);
	m_iMinNeighbor = minNeighNum;
}

void CPaDVC::SetICGNParam(const float deltaP, const int maxIter) {
	m_dDeltaP = deltaP;
	m_iMaxIterationNum = maxIter;

}

/*
	Strategy main funs
*/
void CPaDVC::PiDVCAlgorithm_mul_global_STRATEGY() {

	if (m_eInterpolation == TRICUBIC) {
		//Reject TRICUBIC for globally implementation
		//For using Tricubic interpolation, too much memory required
		cerr << "Globally implementation of DVC is unavailable for TRICUBIC interpolation-!." << endl;
		return;
	}

	//---------------------Memory Allocation-------------------------
	//-! 1. Precomputation memory allocation
	double t1 = omp_get_wtime();
	Precomputation_Prepare_global();
	//-! 2. ICGN memory
	ICGN_Prepare();
	double t_PreICGN_end = omp_get_wtime();
	cout << "----!1.Finish Preparation in " << t_PreICGN_end - t1 << "s" << endl;

	//BSPLINE PREFILTER
	double t_Prefilter_start = omp_get_wtime();
	Prefilter(m_fVolT, (m_fTGBspline[0][0]), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	TransSplineArray();
	PrecomputeGradients_mul_global();
	double t_Precompute_End = omp_get_wtime();
	cout << "----!2-1.Finish Bspline Precomputation in " << t_Precompute_End - t_Prefilter_start << "s" << endl;


	//Using SIFT and kdtree to finish
	Affine_Prepare_kdtree();
	double t_kdtree_end = omp_get_wtime();
	cout << "----!2-2.Finish Building KD-tree in " << t_kdtree_end - t_Precompute_End << "s" << endl;

	//initial guess using SIFT neighbor matches
	ransacCompute.init_random(thread_num);
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); i++)
	{
		int tid = omp_get_thread_num();
		Affine_Compute_kdtree_Global_AUTO_EXPAND(m_POI_global[i]);
	}
#pragma omp barrier
	double t_aff1 = omp_get_wtime();
	cout << "----!3-1.Finish Affine Initial Guess in " << t_aff1 - t_kdtree_end << "s" << endl;

	//ICGN
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); ++i) {
		if (m_POI_global[i].GetEmpty() == 0)
			ICGN_Compute_BSPLINE_Global(m_POI_global[i], omp_get_thread_num(), m_iMaxIterationNum, m_dDeltaP);
		if (i % int(m_POI_global.size() / 10) == 0) {
			cout << "Processing POI " << i / int(m_POI_global.size() / 10) << "0%" << endl;
		}
	}
	double t_ICGN_End = omp_get_wtime();
	cout << "----!4.Finish ICGN in " << t_ICGN_End - t_aff1 << "s" << endl;

	//Strategy 3
	Strategy3_simple_transfer();
	double t_Strategy3_End = omp_get_wtime();
	cout << "----!4.Finish Strategy3 in " << t_Strategy3_End - t_ICGN_End << "s" << endl;

	m_Timer.dPreparationTime = (t_PreICGN_end - t1);
	m_Timer.dPrecomputationTime = (t_Precompute_End - t_Prefilter_start);
	m_Timer.dBuildKDTreeTime = t_kdtree_end - t_Precompute_End;
	m_Timer.dLocalAffineTime = (t_aff1 - t_kdtree_end);
	m_Timer.dICGNTime = (t_ICGN_End - t_aff1);
	m_Timer.dTransferStrategyTime = (t_Strategy3_End - t_ICGN_End);
	m_Timer.dConsumedTime = omp_get_wtime() - t1;
	m_bIsExecuted = true;
}

void CPaDVC::PiDVCAlgorithm_mul_global_FFTCC() {

	if (m_eInterpolation == TRICUBIC) {
		//Reject TRICUBIC for globally implementation
		//For using Tricubic interpolation, too much memory required
		cerr << "Globally implementation of DVC is unavailable for TRICUBIC interpolation-!." << endl;
		return;
	}
	m_bIsExecuted = true;

	//---------------------Memory Allocation-------------------------
	//-! 1. Precomputation memory allocation
	double t1 = omp_get_wtime();
	Precomputation_Prepare_global();
	FFTCC_Prepare(m_iSubsetX, m_iSubsetY, m_iSubsetZ);
	ICGN_Prepare();
	double t_PreICGN_end = omp_get_wtime();
	cout << "----!1.Finish Preparation in " << t_PreICGN_end - t1 << "s" << endl;

	//BSPLINE PREFILTER
	double t_Prefilter_start = omp_get_wtime();
	CMemManager<float>::hCreatePtr(m_fTGBspline, m_iOriginVolDepth, m_iOriginVolHeight, m_iOriginVolWidth);
	Prefilter(m_fVolT, (&m_fTGBspline[0][0][0]), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	TransSplineArray();
	double t_Prefilter_end = omp_get_wtime();
	cout << "----!2.Finish Prefilter in " << t_Prefilter_end - t_Prefilter_start << "s" << endl;
	PrecomputeGradients_mul_global();
	double t_Precompute_End = omp_get_wtime();
	cout << "----!2.Finish Precomputation in " << t_Precompute_End - t_Prefilter_end << "s" << endl;

	//FFT-CC
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); i++)
	{
		int tid = omp_get_thread_num();
		FFTCC_Compute_Global(m_POI_global[i], tid);
		m_POI_global[i].strategy = FFTCC;
	}
#pragma omp barrier
	double t_FFTCc_End = omp_get_wtime();
	cout << "----!3.Finish FFT-CC Initial Guess in " << t_FFTCc_End - t_Precompute_End << "s" << endl;

	//ICGN
	int total_POI_num = m_POI_global.size();
	int tenPercent_num = total_POI_num / 10;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); ++i) {
		int tid = omp_get_thread_num();
		ICGN_Compute_BSPLINE_Global(m_POI_global[i], tid, m_iMaxIterationNum, m_dDeltaP);
		if (tenPercent_num && i%tenPercent_num == 0) {
			cout << "Processing POI " << i / (total_POI_num / 10) << 0 << "%" << endl;
		}
	}
#pragma omp barrier
	double t_ICGN_End = omp_get_wtime();
	cout << "----!4.Finish ICGN in " << t_ICGN_End - t_FFTCc_End << "s" << endl;

	m_Timer.dPreparationTime = (t_PreICGN_end - t1);
	m_Timer.dPrecomputationTime = (t_Precompute_End - t_Prefilter_start);
	m_Timer.dFFTCcTime = t_FFTCc_End - t_Precompute_End;
	m_Timer.dICGNTime = (t_ICGN_End - t_FFTCc_End);
	m_Timer.dConsumedTime = omp_get_wtime() - t1;
}


void CPaDVC::PiDVCAlgorithm_mul_global_ICGN_Only() {

	if (m_eInterpolation == TRICUBIC) {
		//Reject TRICUBIC for globally implementation
		//For using Tricubic interpolation, too much memory required
		cerr << "Globally implementation of DVC is unavailable for TRICUBIC interpolation-!." << endl;
		return;
	}
	m_bIsExecuted = true;

	//---------------------Memory Allocation-------------------------
	//-! 1. Precomputation memory allocation
	double t1 = omp_get_wtime();
	Precomputation_Prepare_global();
	ICGN_Prepare();
	double t_PreICGN_end = omp_get_wtime();
	cout << "----!1.Finish Preparation in " << t_PreICGN_end - t1 << "s" << endl;

	//BSPLINE PREFILTER
	double t_Prefilter_start = omp_get_wtime();
	CMemManager<float>::hCreatePtr(m_fTGBspline, m_iOriginVolDepth, m_iOriginVolHeight, m_iOriginVolWidth);
	Prefilter(m_fVolT, (&m_fTGBspline[0][0][0]), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	TransSplineArray();
	double t_Prefilter_end = omp_get_wtime();
	cout << "----!2.Finish Prefilter in " << t_Prefilter_end - t_Prefilter_start << "s" << endl;
	PrecomputeGradients_mul_global();
	double t_Precompute_End = omp_get_wtime();
	cout << "----!2.Finish Precomputation in " << t_Precompute_End - t_Prefilter_end << "s" << endl;

	//ICGN
	int total_POI_num = m_POI_global.size();
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); ++i) {
		int tid = omp_get_thread_num();
		ICGN_Compute_BSPLINE_Global(m_POI_global[i], tid, m_iMaxIterationNum, m_dDeltaP);
		if (i % (total_POI_num / 10) == 0) {
			cout << "Processing POI " << i / (total_POI_num / 10) << 0 << "%" << endl;
		}
	}
#pragma omp barrier
	double t_ICGN_End = omp_get_wtime();
	cout << "----!4.Finish ICGN in " << t_ICGN_End - t_Precompute_End << "s" << endl;

	m_Timer.dPreparationTime = (t_PreICGN_end - t1);
	m_Timer.dPrecomputationTime = (t_Precompute_End - t_Prefilter_start);
	m_Timer.dICGNTime = (t_ICGN_End - t_Precompute_End);
	m_Timer.dConsumedTime = omp_get_wtime() - t1;
}

/*
	Deatiled computation funs below
*/
//-! 1.Preparation & finalization
void CPaDVC::Precomputation_Prepare_global() {

	//-! 3. Allocate Memory
	CMemManager<float>::hCreatePtr(m_fRx, m_iOriginROIDepth, m_iOriginROIHeight, m_iOriginROIWidth);
	CMemManager<float>::hCreatePtr(m_fRy, m_iOriginROIDepth, m_iOriginROIHeight, m_iOriginROIWidth);
	CMemManager<float>::hCreatePtr(m_fRz, m_iOriginROIDepth, m_iOriginROIHeight, m_iOriginROIWidth);
	CMemManager<float>::hCreatePtr(m_fTGBspline, m_iOriginVolDepth, m_iOriginVolHeight, m_iOriginVolWidth);
	if (m_fRx[0][0] == NULL)
		cerr << "Malloc Memory m_fRx Failed" << endl;
	if (m_fRy[0][0] == NULL)
		cerr << "Malloc Memory m_fRx Failed" << endl;
	if (m_fRz[0][0] == NULL)
		cerr << "Malloc Memory m_fRx Failed" << endl;
	if (m_fTGBspline[0][0] == NULL)
		cerr << "Malloc Memory m_fRx Failed" << endl;


}

void CPaDVC::Precomputation_Finalize()
{
	//-! Destroy the "float" parameters
	CMemManager<float>::hDestroyPtr(m_fVolR);
	CMemManager<float>::hDestroyPtr(m_fVolT);

	if (m_eInterpolation == BSPLINE) {
		CMemManager<float>::hDestroyPtr(m_fTGBspline);
	}

	CMemManager<float>::hDestroyPtr(m_fR);
	CMemManager<float>::hDestroyPtr(m_fRx);
	CMemManager<float>::hDestroyPtr(m_fRy);
	CMemManager<float>::hDestroyPtr(m_fRz);
}

void CPaDVC::Affine_Prepare_kdtree()
{
	KD_Build(kd, kd_idx, Ref_point);
}

void CPaDVC::Affine_Finalize_kdtree()
{
	KD_Destroy(kd, kd_idx);
}

void CPaDVC::FFTCC_Prepare(int iSubsetX, int iSubsetY, int iSubsetZ) {
	//thread_num

	int iFFTSubW = iSubsetX * 2;
	int iFFTSubH = iSubsetY * 2;
	int iFFTSubD = iSubsetZ * 2;
	//int iNumberofPOI1Batch = m_iNumberX*m_iNumberY*m_iNumberZ;

	CMemManager<float>::hCreatePtr(m_fSubset1, thread_num*iFFTSubD*iFFTSubH*iFFTSubW);
	CMemManager<float>::hCreatePtr(m_fSubset2, thread_num*iFFTSubD*iFFTSubH*iFFTSubW);
	CMemManager<float>::hCreatePtr(m_fSubsetC, thread_num*iFFTSubD*iFFTSubH*iFFTSubW);

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

void CPaDVC::FFTCC_Finalize() {
	if (m_fSubset1 != nullptr)
		CMemManager<float>::hDestroyPtr(m_fSubset1);
	if (m_fSubset2 != nullptr)
		CMemManager<float>::hDestroyPtr(m_fSubset2);
	if (m_fSubsetC != nullptr)
		CMemManager<float>::hDestroyPtr(m_fSubsetC);

	if (m_fftwPlan1 != nullptr)
		for (int i = 0; i < thread_num; i++)
		{
			fftwf_destroy_plan(m_fftwPlan1[i]);
			fftwf_destroy_plan(m_fftwPlan2[i]);
			fftwf_destroy_plan(m_rfftwPlan[i]);
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

void CPaDVC::ICGN_Prepare()
{
	int iSubsetW = 2 * m_iSubsetX + 1;
	int iSubsetH = 2 * m_iSubsetY + 1;
	int iSubsetD = 2 * m_iSubsetZ + 1;
	//int iNumofPOI1Batch = m_iNumberX * m_iNumberY * m_iNumberZ;

	CMemManager<float>::hCreatePtr(m_fRDescent, thread_num, iSubsetD, iSubsetH, iSubsetW, 12);
	CMemManager<float>::hCreatePtr(m_fSubsetR, thread_num, iSubsetD, iSubsetH, iSubsetW);
	CMemManager<float>::hCreatePtr(m_fSubsetT, thread_num, iSubsetD, iSubsetH, iSubsetW);
}

void CPaDVC::ICGN_Finalize()
{
	CMemManager<float>::hDestroyPtr(m_fSubsetR);
	CMemManager<float>::hDestroyPtr(m_fSubsetT);
	CMemManager<float>::hDestroyPtr(m_fRDescent);
}

//-! 2.Precomputation
void CPaDVC::PrecomputeGradients_mul_global()
/*
Precompute the voxel gradients using the central difference
scheme.
*/
{
#pragma omp parallel for
	for (int i = 0; i < m_iOriginROIDepth; i++)
	{
		for (int j = 0; j < m_iOriginROIHeight; j++)
		{
			for (int k = 0; k < m_iOriginROIWidth; k++)
			{
				m_fRx[i][j][k] = 0.5f*(m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth, k + 2, j + 1, i + 1)] - m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth, k, j + 1, i + 1)]);
				m_fRy[i][j][k] = 0.5f*(m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth, k + 1, j + 2, i + 1)] - m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth, k + 1, j, i + 1)]);
				m_fRz[i][j][k] = 0.5f*(m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth, k + 1, j + 1, i + 2)] - m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth, k + 1, j + 1, i)]);
			}
		}
	}
}

void CPaDVC::TransSplineArray() {
	//Trans to another array with special margin
	float ***New_fTGBspline;
	CMemManager<float>::hCreatePtr(New_fTGBspline, m_iOriginVolDepth + 4, m_iOriginVolHeight + 4, m_iOriginVolWidth + 4);

	int z, y, x;
	for (int i = 0; i < m_iOriginVolDepth + 4; ++i) {
		z = MIN(MAX(i - 2, 0), m_iOriginVolDepth - 1);

		for (int j = 0; j < m_iOriginVolHeight + 4; ++j) {
			y = MIN(MAX(j - 2, 0), m_iOriginVolHeight - 1);

			for (int k = 0; k < m_iOriginVolWidth + 4; ++k) {
				x = MIN(MAX(k - 2, 0), m_iOriginVolWidth - 1);

				New_fTGBspline[i][j][k] = m_fTGBspline[z][y][x];
			}
		}
	}

	CMemManager<float>::hDestroyPtr(m_fTGBspline);
	m_fTGBspline = New_fTGBspline;
}

 //-! 3.Compute
 //-! 3-1.Computation Tools funs
void CPaDVC::KD_Build(kdtree *&_kd, int *&_idx, const vector<Cvec> &vc) {

	//Construct the kd_tree
	_kd = kd_create(3);

	int kp_size = vc.size();
	_idx = new int[kp_size];
	for (int i = 0; i < kp_size; i++)
	{
		_idx[i] = i;
		double temp[3] = { static_cast<double>(vc[i].x),
			static_cast<double>(vc[i].y),
			static_cast<double>(vc[i].z) };

		kd_insert(_kd, temp, &_idx[i]);
	}

}

void CPaDVC::KD_Destroy(kdtree *&_kd, int *&_idx) {
	if (_idx)
		delete[]_idx;
	if (_kd)
		kd_free(_kd);;
	_kd = nullptr;
	_idx = nullptr;
}

void CPaDVC::KD_RangeSerach(kdtree* _kd, vector<Cvec> &neighbor, vector<int> &idx, const Cvec query, const double range) {
	if (range<0) {
		cerr << "Warning for searching negative range neighbors" << endl;
		return;
	}
	double point[3] = { query.x, query.y, query.z };
	kdres *result = kd_nearest_range(_kd, point, range);
	while (!kd_res_end(result)) {
		double pos[3];
		kd_res_item(result, pos);
		void *data = kd_res_item_data(result);
		int data_int = *(static_cast<int*>(data));
		neighbor.push_back(Cvec(pos[0], pos[1], pos[2]));
		idx.push_back(data_int);

		kd_res_next(result);
	}
	kd_res_free(result);
	return;
}

void CPaDVC::KD_KNNSerach(kdtree* _kd, vector<Cvec> &neighbor, vector<int> &idx, const Cvec query, const int K) {
	double point[3] = { query.x, query.y, query.z };
	kdres *result = kd_nearest_n(_kd, point, K);
	double max_dist_square = 0.0;

	while (!kd_res_end(result)) {
		double pos[3];
		kd_res_item(result, pos);
		void *data = kd_res_item_data(result);
		int data_int = *(static_cast<int*>(data));
		neighbor.push_back(Cvec(pos[0], pos[1], pos[2]));
		idx.push_back(data_int);

		kd_res_next(result);
	}
	kd_res_free(result);
	return;
}

int CPaDVC::Collect_neighborPOI(CPOI &POI_, int *Neighbour_id) {
	int num = 0;
	for (int i = 0; i<6; ++i) {
		int id = POI_.neighbor[i];
		if (id >= 0 && m_POI_global[id].GetProcessed())
			Neighbour_id[num++] = id;
	}
	return num;
}

int CPaDVC::Collect_neighborBestPOI(CPOI &POI_, int *Neighbour_id) {
	int num = 0;
	float max = -1.1 ;
	for (int i = 0; i<6; ++i) {
		int id = POI_.neighbor[i];
		if (id >= 0 && m_POI_global[id].GetProcessed()) {
			if (m_POI_global[id].ZNCC() > max) {
				Neighbour_id[0] = id;
				max = m_POI_global[id].ZNCC();
				++num;
			}
		}
	}
	return num;
}

void CPaDVC::BatchICGN(vector<int> toBe, int tnum) {

#pragma omp parallel for num_threads(tnum)
	for (int i = 0; i < toBe.size(); ++i) {
		int tid = omp_get_thread_num();
		int poi_id = toBe[i];
		ICGN_Compute_BSPLINE_Global(m_POI_global[poi_id], tid, m_iMaxIterationNum, m_dDeltaP);
	}
	return;
}

int CPaDVC::InverseHessian_GaussianJordan(vector<vector<float>>&m_fInvHessian,
	vector<vector<float>>&m_fHessian)
{
	int k, l, m, n;
	int iTemp;
	float dTemp;

	for (l = 0; l < 12; l++)
	{
		for (m = 0; m < 12; m++)
		{
			if (l == m)
				m_fInvHessian[l][m] = 1;
			else
				m_fInvHessian[l][m] = 0;
		}
	}

	for (l = 0; l < 12; l++)
	{
		//-! Find pivot (maximum lth column element) in the rest (6-l) rows
		iTemp = l;
		for (m = l + 1; m < 12; m++)
		{
			if (m_fHessian[m][l] > m_fHessian[iTemp][l])
			{
				iTemp = m;
			}
		}
		if (fabs(m_fHessian[iTemp][l]) == 0)
		{
			return 1;
		}
		//-! Swap the row which has maximum lth column element
		if (iTemp != l)
		{
			for (k = 0; k < 12; k++)
			{
				dTemp = m_fHessian[l][k];
				m_fHessian[l][k] = m_fHessian[iTemp][k];
				m_fHessian[iTemp][k] = dTemp;

				dTemp = m_fInvHessian[l][k];
				m_fInvHessian[l][k] = m_fInvHessian[iTemp][k];
				m_fInvHessian[iTemp][k] = dTemp;
			}
		}
		//-! Perform row operation to form required identity matrix out of the Hessian matrix
		for (m = 0; m < 12; m++)
		{
			dTemp = m_fHessian[m][l];
			if (m != l)
			{
				for (n = 0; n < 12; n++)
				{
					m_fInvHessian[m][n] -= m_fInvHessian[l][n] * dTemp / m_fHessian[l][l];
					m_fHessian[m][n] -= m_fHessian[l][n] * dTemp / m_fHessian[l][l];
				}
			}
			else
			{
				for (n = 0; n < 12; n++)
				{
					m_fInvHessian[m][n] /= dTemp;
					m_fHessian[m][n] /= dTemp;
				}
			}
		}
	}
	return 0;
}

 //-! 3-2.Computation funs
 void CPaDVC::FFTCC_Compute_Global(CPOI &POI_, int iID_) {
 	//StopWatchWin fftccWatch;
 	POI_.SetProcessed(1);
 
 	int iFFTSubW = m_iSubsetX * 2;
 	int iFFTSubH = m_iSubsetY * 2;
 	int iFFTSubD = m_iSubsetZ * 2;
 	int iFFTSize = iFFTSubW * iFFTSubH * iFFTSubD;
 	int iFFTFreqSize = iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1);
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

 	// cout << m_iU << ", " << m_iV << ", " << m_iW << endl;
 
 	// Stop the timer for FFT-CC algorithm and calculate the time consumed
 	//fftccWatch.stop();
 	//POI_.SetFFTCCTime(fftccWatch.getTime()); //unit: millisec
 }
 

void CPaDVC::Affine_Compute_kdtree_Global_AUTO_EXPAND(CPOI &POI_) {

	//Get global coordinate
	int x = POI_.G_X();
	int y = POI_.G_Y();
	int z = POI_.G_Z();

#ifdef _DEBUG
	if (x == debugP[0] && y == debugP[1] && z == debugP[2]) {
		debugP[0] = debugP[0];
		cerr << "catched!" << endl;
	}
	else {
		return;
	}
#endif 

	//Set range
	double range = sqrt(m_iSubsetX*m_iSubsetX + m_iSubsetY*m_iSubsetY + m_iSubsetZ*m_iSubsetZ) + 0.01;//experiment
	int enough = 1;

	//transform the result into vector of cvec to get the affine transformation matrix and initial guess
	vector<Cvec> tmp_ref, tmp_tar;
	vector<int> tmp_index;
	
	KD_RangeSerach(kd, tmp_ref, tmp_index, Cvec(x,y,z), range);
	for(auto i: tmp_index)
		tmp_tar.push_back(Tar_point[i]);
	TransferLocalCoor(tmp_ref, Cvec(x,y,z));
	TransferLocalCoor(tmp_tar, Cvec(x,y,z));

	if (tmp_ref.size() >= m_iMinNeighbor) {
		// Examine whether Inside SubVolume Enough
		vector<Cvec> tmp_inside_ref, tmp_inside_tar;
		vector<int> tmp_inside_idx;
		float x_border = static_cast<float>((m_iSubsetX)) + EPSILON;
		float y_border = static_cast<float>((m_iSubsetY)) + EPSILON;
		float z_border = static_cast<float>((m_iSubsetZ)) + EPSILON;
		for (int i = 0; i < tmp_index.size(); ++i) {
			//inside the subvolume
			//coordinate of tmp_ref is local, hence not subtraction
			if (fabsf(tmp_ref[i].x) < x_border &&
				fabsf(tmp_ref[i].y) < y_border &&
				fabsf(tmp_ref[i].z) < z_border ){
				//put KPs inside subvolume into temporary vectors.
				tmp_inside_ref.push_back(tmp_ref[i]);
				tmp_inside_tar.push_back(tmp_tar[i]);
				tmp_inside_idx.push_back(tmp_index[i]);
			}
		}
		if (tmp_inside_ref.size() >= m_iMinNeighbor) {
			//using KPs inside the box when enough 
			tmp_ref = tmp_inside_ref;
			tmp_tar = tmp_inside_tar;
			tmp_index = tmp_inside_idx;
			POI_.SetStrategy(Search_Subset);
		}
		else {
			// otherwise, using KPs of the circumscribed ball
			POI_.SetSearchRadius(range);
			POI_.SetStrategy(Search_Radius);
		}
	} else {
		// Not Enough KP in subvolume or circumscribed ball
		// Auto extending
		// KNN search is performed
		tmp_ref.clear();
		tmp_tar.clear();
		tmp_index.clear();
		const float up_radius_square = m_iExpandRatio*m_iExpandRatio*range*range;

		KD_KNNSerach(kd, tmp_ref, tmp_index, Cvec(x,y,z), m_iMinNeighbor);
		TransferLocalCoor(tmp_ref, Cvec(x,y,z));
		float maxDist = maxDistanceSquareTo(tmp_ref, Cvec(0,0,0));

		if (maxDist > up_radius_square) {
			//searching out of max range
			enough = 0;
			POI_.SetRangeGood(enough);
			POI_.SetEmpty(1);
			return;
		} else {
			for(auto i: tmp_index)
				tmp_tar.push_back(Tar_point[i]);
			TransferLocalCoor(tmp_tar, Cvec(x,y,z));
			POI_.SetSearchRadius(sqrtf(maxDist));
			POI_.SetStrategy(Search_Expand_Radius);
		}

	}

	if (tmp_ref.size() < 4)
		return;
	//store candidate key point pairs
	POI_.c_ref = tmp_ref;
	POI_.c_tar = tmp_tar;
	POI_.s_idx = tmp_index;
	POI_.num_candidate = tmp_ref.size();

	ransacCompute.Ransac(tmp_ref, tmp_tar, POI_.m_fAffine, POI_.f_ref, POI_.f_tar);

	vector<float>tempP0(12, 0);
	// U V W 
	tempP0[0] = POI_.m_fAffine[9];
	tempP0[4] = POI_.m_fAffine[10];
	tempP0[8] = POI_.m_fAffine[11];

	// Ux Uy Uz
	tempP0[1] = POI_.m_fAffine[0] - 1;
	tempP0[2] = POI_.m_fAffine[3];
	tempP0[3] = POI_.m_fAffine[6];

	// Vx Vy Vz
	tempP0[5] = POI_.m_fAffine[1];
	tempP0[6] = POI_.m_fAffine[4] - 1;
	tempP0[7] = POI_.m_fAffine[7];

	// Wx Wy Wz
	tempP0[9]  = POI_.m_fAffine[2];
	tempP0[10] = POI_.m_fAffine[5];
	tempP0[11] = POI_.m_fAffine[8] - 1;
	
	POI_.SetP0(tempP0);
	POI_.SetRangeGood(enough);
	POI_.SetEmpty(0);
	POI_.num_final = POI_.f_ref.size();

	tmp_ref.clear();
	tmp_tar.clear();
	tmp_ref.shrink_to_fit();
	tmp_tar.shrink_to_fit();
}

void CPaDVC::ICGN_Compute_BSPLINE_Global(CPOI &POI_, int iID_, int iMaxIteration, float fDeltaP)
{
//#ifdef _DEBUG
//	if (POI_.G_X() == debugP[0] && POI_.G_Y() == debugP[1] && POI_.G_Z() == debugP[2]) {
//		debugP[0] = debugP[0];
//		cerr << "catched! IC-GN" << endl;
//	}
//	else {
//		return;
//	}
//#endif 

	//StopWatchWin w;
	POI_.SetProcessed(2);

	//-! Define the size of subvolume window for IC-GN algorithm
	int iSubsetX = m_iSubsetX;
	int iSubsetY = m_iSubsetY;
	int iSubsetZ = m_iSubsetZ;
	int iOriginROIWidth = m_iOriginROIWidth;
	int iOriginROIHeight = m_iOriginROIHeight;
	int iOriginROIDepth = m_iOriginROIDepth;
	int iOriginVolWidth = m_iOriginVolWidth;
	int iOriginVolHeight = m_iOriginVolHeight;
	int iOriginVolDepth = m_iOriginVolDepth;
	int iSubsetW = iSubsetX * 2 + 1;
	int iSubsetH = iSubsetY * 2 + 1;
	int iSubsetD = iSubsetZ * 2 + 1;
	float fSubsetSize = float(iSubsetD*iSubsetH*iSubsetW);
	float ***m_fSubsetR_id = m_fSubsetR[iID_];
	float ***m_fSubsetT_id = m_fSubsetT[iID_];
	float ****m_fRDescent_id = m_fRDescent[iID_];

	int iTempX, iTempY, iTempZ;
	float fSubAveR, fSubAveT, fSubNorR, fSubNorT;
	float fU, fUx, fUy, fUz, fV, fVx, fVy, fVz, fW, fWx, fWy, fWz;
	float fDU, fDUx, fDUy, fDUz, fDV, fDVx, fDVy, fDVz, fDW, fDWx, fDWy, fDWz;
	float fWarpX, fWarpY, fWarpZ, m_fTemp, m_fTempX, m_fTempY, m_fTempZ;
	float fErrors;
	float fZNCC;
	float f00, f10, f20;
	float f01, f11, f21;
	float f02, f12, f22;
	float f03, f13, f23;
	vector<float>P(12, 0);
	vector<float>DP(12, 0);

	vector<float> m_fNumerator(12, 0);
	// 12-vector
	vector<vector<float>> m_fInvHessian(12, vector<float>(12, 0));
	// Inverse Hessian matrix 12x12
	vector<vector<float>> m_fHessian(12, vector<float>(12, 0));
	// Hessian of a POI 12x12
	vector<vector<float>> m_fHessianXYZ(12, vector<float>(12, 0));
	// Hessian of each point around a POI 12x12
	vector<vector<float>> m_fJacobian(3, vector<float>(12, 0));
	// Jacobian matrix deltaR*(dW/dP) 3x12
	vector<vector<float>> m_fWarp(4, vector<float>(4, 0));
	// Warp Function 4x4

	float w_x[4];
	float w_y[4];
	float w_z[4];
	float sum_x[4];
	float sum_y[4];
	//BSPLINE

	//-! Start the timer for IC-GN algorithm

	//w.start();


	//-! Initialize parameters of Subvolume R
	fSubAveR = 0; // R_m
	fSubNorR = 0; // sqrt (Sigma(R_i - R_m)^2)
				  //-! Initialize the Hessian matrix for each subvolume
	for (int k = 0; k < 12; k++)
	{
		for (int n = 0; n < 12; n++)
		{
			m_fHessian[k][n] = 0;
		}
	}


	//-! Feed the gray intensity to subvolume R
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			for (int n = 0; n < iSubsetW; n++)
			{
				m_fSubsetR_id[l][m][n] =
					m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth,
						POI_.G_X() - iSubsetX + n,
						POI_.G_Y() - iSubsetY + m,
						POI_.G_Z() - iSubsetZ + l)];

				fSubAveR += m_fSubsetR_id[l][m][n];

				//-! Evaluate the Jacbian dW/dp at (x, 0), etc. POI
				m_fJacobian[0][0] = 1;
				m_fJacobian[0][1] = float(n - iSubsetX);
				m_fJacobian[0][2] = float(m - iSubsetY);
				m_fJacobian[0][3] = float(l - iSubsetZ);
				m_fJacobian[0][4] = 0;
				m_fJacobian[0][5] = 0;
				m_fJacobian[0][6] = 0;
				m_fJacobian[0][7] = 0;
				m_fJacobian[0][8] = 0;
				m_fJacobian[0][9] = 0;
				m_fJacobian[0][10] = 0;
				m_fJacobian[0][11] = 0;

				m_fJacobian[1][0] = 0;
				m_fJacobian[1][1] = 0;
				m_fJacobian[1][2] = 0;
				m_fJacobian[1][3] = 0;
				m_fJacobian[1][4] = 1;
				m_fJacobian[1][5] = float(n - iSubsetX);
				m_fJacobian[1][6] = float(m - iSubsetY);
				m_fJacobian[1][7] = float(l - iSubsetZ);
				m_fJacobian[1][8] = 0;
				m_fJacobian[1][9] = 0;
				m_fJacobian[1][10] = 0;
				m_fJacobian[1][11] = 0;

				m_fJacobian[2][0] = 0;
				m_fJacobian[2][1] = 0;
				m_fJacobian[2][2] = 0;
				m_fJacobian[2][3] = 0;
				m_fJacobian[2][4] = 0;
				m_fJacobian[2][5] = 0;
				m_fJacobian[2][6] = 0;
				m_fJacobian[2][7] = 0;
				m_fJacobian[2][8] = 1;
				m_fJacobian[2][9] = float(n - iSubsetX);
				m_fJacobian[2][10] = float(m - iSubsetY);
				m_fJacobian[2][11] = float(l - iSubsetZ);

				float *m_fRDescent_id_ilmn = m_fRDescent_id[l][m][n];
				//-! Compute the steepest descent image DealtR*dW/dp
				for (int k = 0; k < 12; k++)
				{
					//m_fRDescent_id[l][m][n][k] =
					m_fRDescent_id_ilmn[k] =
						m_fRx[POI_.Z() - iSubsetZ + l][POI_.Y() - iSubsetY + m][POI_.X() - iSubsetX + n] * m_fJacobian[0][k] +
						m_fRy[POI_.Z() - iSubsetZ + l][POI_.Y() - iSubsetY + m][POI_.X() - iSubsetX + n] * m_fJacobian[1][k] +
						m_fRz[POI_.Z() - iSubsetZ + l][POI_.Y() - iSubsetY + m][POI_.X() - iSubsetX + n] * m_fJacobian[2][k];
				}

				//-! Compute the Hessian matrix
				for (int j = 0; j < 12; j++)
				{
					//float m_fRDescent_lmnj_tmp = m_fRDescent_id[l][m][n][j];
					for (int k = 0; k < 12; k++)
					{
						//-! Hessian matrix at each point
						//Origin Accessing
						//m_fHessianXYZ[j][k] = m_fRDescent_id[l][m][n][j] * m_fRDescent_id[l][m][n][k];
						m_fHessianXYZ[j][k] = m_fRDescent_id_ilmn[j] * m_fRDescent_id_ilmn[k];

						//-! sum of Hessian matrix at all the points in subvolume R
						m_fHessian[j][k] += m_fHessianXYZ[j][k];
					}
				}
			}
		}
	}

	//DEBUG
	POI_.GrayValue = m_fSubsetR_id[iSubsetZ][iSubsetY][iSubsetX];
	//cout << POI_.G_X() << ","<< POI_.G_Y() << "," << POI_.G_Z() << "," << POI_.GrayValue << endl;

	//-! Check if Subset R is a all dark subvolume
	if (fSubAveR == 0)
	{
		POI_.SetDarkSubset(2);
		POI_.SetProcessed(-1);
		return;
	}
	fSubAveR /= fSubsetSize;
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			for (int n = 0; n < iSubsetW; n++)
			{
				m_fSubsetR_id[l][m][n] = m_fSubsetR_id[l][m][n] - fSubAveR;	// R_i - R_m
				fSubNorR += pow(m_fSubsetR_id[l][m][n], 2);				// Sigma(R_i - R_m)^2
			}
		}
	}
	fSubNorR = sqrt(fSubNorR);	 // sqrt (Sigma(R_i - R_m)^2)
	if (fSubNorR == 0)
	{
		POI_.SetDarkSubset(3);
		POI_.SetProcessed(-1);
		return;
	}

	//-! Invert the Hessian matrix (Gauss-Jordan algorithm)
	if (1 == InverseHessian_GaussianJordan(m_fInvHessian, m_fHessian))
	{
		POI_.SetInvertibleHessian(1);
		POI_.SetProcessed(-1);
		return;
	}

	//-! Initialize matrix P and DP
	POI_.SetP(POI_.P0());
	fU = POI_.P()[0];
	fUx = POI_.P()[1];
	fUy = POI_.P()[2];
	fUz = POI_.P()[3];
	fV = POI_.P()[4];
	fVx = POI_.P()[5];
	fVy = POI_.P()[6];
	fVz = POI_.P()[7];
	fW = POI_.P()[8];
	fWx = POI_.P()[9];
	fWy = POI_.P()[10];
	fWz = POI_.P()[11];

	//-! Initialize the warp matrix: W
	m_fWarp[0][0] = 1 + fUx;		m_fWarp[0][1] = fUy;		m_fWarp[0][2] = fUz;		m_fWarp[0][3] = fU;
	m_fWarp[1][0] = fVx;			m_fWarp[1][1] = 1 + fVy;	m_fWarp[1][2] = fVz;		m_fWarp[1][3] = fV;
	m_fWarp[2][0] = fWx;			m_fWarp[2][1] = fWy;		m_fWarp[2][2] = 1 + fWz;	m_fWarp[2][3] = fW;
	m_fWarp[3][0] = 0;				m_fWarp[3][1] = 0;			m_fWarp[3][2] = 0;			m_fWarp[3][3] = 1;

	//-! Initialize DeltaP
	fDU = fDUx = fDUy = fDUz = fDV = fDVx = fDVy = fDVz = fDW = fDWx = fDWy = fDWz = 0;

	//-! Filled warped voxel into Subset T
	fSubAveT = 0;
	fSubNorT = 0;
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
			for (int n = 0; n < iSubsetW; n++)
			{
				//-! Calculate the location of warped subvolume T
				fWarpX =
					POI_.X() + 1 + m_fWarp[0][0] * (n - iSubsetX) + m_fWarp[0][1] * (m - iSubsetY) + m_fWarp[0][2] * (l - iSubsetZ) + m_fWarp[0][3];
				fWarpY =
					POI_.Y() + 1 + m_fWarp[1][0] * (n - iSubsetX) + m_fWarp[1][1] * (m - iSubsetY) + m_fWarp[1][2] * (l - iSubsetZ) + m_fWarp[1][3];
				fWarpZ =
					POI_.Z() + 1 + m_fWarp[2][0] * (n - iSubsetX) + m_fWarp[2][1] * (m - iSubsetY) + m_fWarp[2][2] * (l - iSubsetZ) + m_fWarp[2][3];
				iTempX = int(fWarpX);
				iTempY = int(fWarpY);
				iTempZ = int(fWarpZ);

				m_fTempX = fWarpX - iTempX;
				m_fTempY = fWarpY - iTempY;
				m_fTempZ = fWarpZ - iTempZ;
				// To make sure BSPline coefficient access is valid and avoid too many judges at interpolation
				if ((iTempX >= 1) && (iTempY >= 1) && (iTempZ >= 1) &&
					(iTempX <= (iOriginVolWidth - 3)) && (iTempY <= (iOriginVolHeight - 3)) && (iTempZ <= (iOriginVolDepth-3)))
				{

					//-! If it is integer-pixel location, feed the gray intensity of T into the subvolume T
					if (m_fTempX == 0 && m_fTempY == 0 && m_fTempZ == 0)
						m_fSubsetT_id_lm[n] = m_fVolT[ELT(m_iOriginVolHeight, m_iOriginVolWidth, iTempX , iTempY , iTempZ )];

					//-! In most case, it is sub-pixel location, estimate the gary intensity using interpolation
					else {

						w_x[0] = w0_c(m_fTempX);
						w_x[1] = w1_c(m_fTempX);
						w_x[2] = w2_c(m_fTempX);
						w_x[3] = w3_c(m_fTempX);

						w_y[0] = w0_c(m_fTempY);
						w_y[1] = w1_c(m_fTempY);
						w_y[2] = w2_c(m_fTempY);
						w_y[3] = w3_c(m_fTempY);

						w_z[0] = w0_c(m_fTempZ);
						w_z[1] = w1_c(m_fTempZ);
						w_z[2] = w2_c(m_fTempZ);
						w_z[3] = w3_c(m_fTempZ);

						m_fSubsetT_id_lm[n] = 0;


						//z
						for (int j = 0; j<4; ++j) {
							//y
							//due to 2 margin data point are prepared in the array, thus +2 is necessary
							float **m_fTGBspline_j = m_fTGBspline[iTempZ + 2 - 1 + j];
							for (int i = 0; i<4; ++i) {
								float * m_fTGBspline_ji = m_fTGBspline_j[iTempY + 2 - 1 + i];
								/*sum_x[i] =
									w_x[0] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 - 1]
									+ w_x[1] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1]
									+ w_x[2] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 1]
									+ w_x[3] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 2];*/
								sum_x[i] =
									  w_x[0] * m_fTGBspline_ji[iTempX + 2 - 1]
									+ w_x[1] * m_fTGBspline_ji[iTempX + 2	 ]
									+ w_x[2] * m_fTGBspline_ji[iTempX + 2 + 1]
									+ w_x[3] * m_fTGBspline_ji[iTempX + 2 + 2];
							}
							sum_y[j] =
								w_y[0] * sum_x[0]
								+ w_y[1] * sum_x[1]
								+ w_y[2] * sum_x[2]
								+ w_y[3] * sum_x[3];
						}

						m_fSubsetT_id_lm[n] =
							w_z[0] * sum_y[0]
							+ w_z[1] * sum_y[1]
							+ w_z[2] * sum_y[2]
							+ w_z[3] * sum_y[3];

					}
					fSubAveT += m_fSubsetT_id_lm[n];

					//debug
#ifdef _JR_DEBUG
					if (((l*iSubsetH + m)*iSubsetW + n) == 0) {
						printf("id:%d; PRef:(%d,%d,%d); Ptar:(%f,%f,%f); val:%f; iTempX:(%d,%d,%d); inerp:(%f)\n", ((l*iSubsetH + m)*iSubsetW + n), n, m, l, fWarpX, fWarpY, fWarpZ, m_fSubsetT_id_lm[n],iTempX, iTempY, iTempZ, m_fTGBspline[iTempZ + 2][iTempY+2][iTempX+2]);
					}
#endif
				}
				else
				{
					POI_.SetOutofROI(1);	// if the loacation of the warped subvolume T is out of the ROI, stop iteration and set p as the current value
					POI_.SetProcessed(-2);
					return;
				}
			}
		}
	}
	fSubAveT /= fSubsetSize;
	//-! Check if Subset T is a all dark subvolume
	if (fSubAveT == 0)
	{
		POI_.SetDarkSubset(4);
		POI_.SetProcessed(-1);
		return;
	}
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			for (int n = 0; n < iSubsetW; n++)
			{
				m_fSubsetT_id[l][m][n] = m_fSubsetT_id[l][m][n] - fSubAveT;	//T_i - T_m
				fSubNorT += pow(m_fSubsetT_id[l][m][n], 2);				// Sigma(T_i - T_m)^2
			}
		}
	}
	fSubNorT = sqrt(fSubNorT);	// sqrt (Sigma(T_i - T_m)^2)

	if (fSubNorT == 0)
	{
		POI_.SetDarkSubset(5);	// Set flag
		POI_.SetProcessed(-1);
		return;
	}

	//-! Compute the error image
	for (int k = 0; k < 12; k++)
	{
		m_fNumerator[k] = 0;
	}
	fZNCC = 0;
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
			float *m_fSubsetR_id_lm = m_fSubsetR_id[l][m];
			for (int n = 0; n < iSubsetW; n++)
			{
				fErrors = (fSubNorR / fSubNorT) * m_fSubsetT_id_lm[n] - m_fSubsetR_id_lm[n];
				fZNCC += m_fSubsetT_id_lm[n] * m_fSubsetR_id_lm[n];
				//-! Compute the numerator
				float *m_fRDescent_id_lmn = m_fRDescent_id[l][m][n];
				for (int k = 0; k < 12; k++)
				{
					//m_fNumerator[k] += (m_fRDescent_id[l][m][n][k] * fErrors);
					m_fNumerator[k] += (m_fRDescent_id_lmn[k] * fErrors);
				}
			}
		}
	}
	fZNCC = fZNCC / (fSubNorT*fSubNorR);
	POI_.SetZNCC(fZNCC);

#ifdef _JR_DEBUG
	printf("iteration%d, normT:%f, normR/normT:%f\n", POI_.Iteration(), fSubNorT, fSubNorR / fSubNorT);
#endif

	//-! Compute DeltaP
	for (int k = 0; k < 12; k++)
	{
		DP[k] = 0;
		for (int n = 0; n < 12; n++)
		{
			DP[k] += (m_fInvHessian[k][n] * m_fNumerator[n]);
		}
	}
	POI_.SetDP(DP);
	fDU = POI_.DP()[0];
	fDUx = POI_.DP()[1];
	fDUy = POI_.DP()[2];
	fDUz = POI_.DP()[3];
	fDV = POI_.DP()[4];
	fDVx = POI_.DP()[5];
	fDVy = POI_.DP()[6];
	fDVz = POI_.DP()[7];
	fDW = POI_.DP()[8];
	fDWx = POI_.DP()[9];
	fDWy = POI_.DP()[10];
	fDWz = POI_.DP()[11];

	//-! Update the warp
	//-! m_fTemp: store the denominator of the Inverse W
	m_fTemp =
		((fDWz + 1)*fDVy + (-fDWy*fDVz + (fDWz + 1)))*fDUx +
		(((-fDWz - 1)*fDVx + fDWx*fDVz)*fDUy + ((fDWy*fDVx + (-fDWx*fDVy - fDWx))*fDUz +
		((fDWz + 1)*fDVy + (-fDWy*fDVz + (fDWz + 1)))));
	if (m_fTemp == 0)
	{
		POI_.SetInvertibleHessian(2);
		POI_.SetProcessed(-1);
		return;
	}

	//-! W(P) <- W(P) o W(DP)^-1
	f00 = (fDWz + 1)*fDVy + (-fDWy*fDVz + (fDWz + 1));
	f10 = (-fDWz - 1)*fDVx + fDWx*fDVz;
	f20 = fDWy*fDVx + (-fDWx*fDVy - fDWx);
	f01 = (-fDWz - 1)*fDUy + fDWy*fDUz;
	f11 = (fDWz + 1)*fDUx + (-fDWx*fDUz + (fDWz + 1));
	f21 = -fDWy*fDUx + (fDWx*fDUy - fDWy);
	f02 = fDVz*fDUy + (-fDVy - 1)*fDUz;
	f12 = -fDVz*fDUx + (fDVx*fDUz - fDVz);
	f22 = (fDVy + 1)*fDUx + (-fDVx*fDUy + (fDVy + 1));
	f03 = (-fDW*fDVz + (fDWz + 1)*fDV)*fDUy + ((fDW*fDVy + (-fDWy*fDV + fDW))*fDUz + ((-fDWz - 1)*fDVy + (fDWy*fDVz + (-fDWz - 1)))*fDU);
	f13 = (fDW*fDVz + (-fDWz - 1)*fDV)*fDUx + ((-fDW*fDVx + fDWx*fDV)*fDUz + (((fDWz + 1)*fDVx - fDWx*fDVz)*fDU + (fDW*fDVz + (-fDWz - 1)*fDV)));
	f23 = (-fDW*fDVy + (fDWy*fDV - fDW))*fDUx + ((fDW*fDVx - fDWx*fDV)*fDUy + ((-fDWy*fDVx + (fDWx*fDVy + fDWx))*fDU + (-fDW*fDVy + (fDWy*fDV - fDW))));
	m_fWarp[0][0] = ((1 + fUx)*f00 + fUy*f10 + fUz*f20) / m_fTemp;	m_fWarp[0][1] = ((1 + fUx)*f01 + fUy*f11 + fUz*f21) / m_fTemp; 	m_fWarp[0][2] = ((1 + fUx)*f02 + fUy*f12 + fUz*f22) / m_fTemp; 	m_fWarp[0][3] = ((1 + fUx)*f03 + fUy*f13 + fUz*f23) / m_fTemp + fU;
	m_fWarp[1][0] = (fVx*f00 + (1 + fVy)*f10 + fVz*f20) / m_fTemp;	m_fWarp[1][1] = (fVx*f01 + (1 + fVy)*f11 + fVz*f21) / m_fTemp;	m_fWarp[1][2] = (fVx*f02 + (1 + fVy)*f12 + fVz*f22) / m_fTemp;	m_fWarp[1][3] = (fVx*f03 + (1 + fVy)*f13 + fVz*f23) / m_fTemp + fV;
	m_fWarp[2][0] = (fWx*f00 + fWy*f10 + (1 + fWz)*f20) / m_fTemp;	m_fWarp[2][1] = (fWx*f01 + fWy*f11 + (1 + fWz)*f21) / m_fTemp;	m_fWarp[2][2] = (fWx*f02 + fWy*f12 + (1 + fWz)*f22) / m_fTemp;	m_fWarp[2][3] = (fWx*f03 + fWy*f13 + (1 + fWz)*f23) / m_fTemp + fW;
	m_fWarp[3][0] = 0;	m_fWarp[3][1] = 0;	m_fWarp[3][2] = 0;	m_fWarp[3][3] = 1;

	//-! Update P
	P[0] = m_fWarp[0][3];	P[1] = m_fWarp[0][0] - 1;	P[2] = m_fWarp[0][1];		P[3] = m_fWarp[0][2];
	P[4] = m_fWarp[1][3];	P[5] = m_fWarp[1][0];		P[6] = m_fWarp[1][1] - 1;	P[7] = m_fWarp[1][2];
	P[8] = m_fWarp[2][3];	P[9] = m_fWarp[2][0];		P[10] = m_fWarp[2][1];		P[11] = m_fWarp[2][2] - 1;
	POI_.SetP(P);
	fU = POI_.P()[0];
	fUx = POI_.P()[1];
	fUy = POI_.P()[2];
	fUz = POI_.P()[3];
	fV = POI_.P()[4];
	fVx = POI_.P()[5];
	fVy = POI_.P()[6];
	fVz = POI_.P()[7];
	fW = POI_.P()[8];
	fWx = POI_.P()[9];
	fWy = POI_.P()[10];
	fWz = POI_.P()[11];
	POI_.SetIteration(1);

#ifdef _JR_DEBUG
	printf("iteration: %d, deltaP:%f\n", 0, sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)));
	printf("\t\tmatrix:%f %f %f %f, %f %f %f %f, %f %f %f %f \n",
		P[0], P[1], P[2], P[3],
		P[4], P[5], P[6], P[7],
		P[8], P[9], P[10], P[11]);
#endif

	//-! Perform interative optimization, with pre-set maximum iteration step
	while (POI_.Iteration() < iMaxIteration && sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)) > fDeltaP)
	{
		POI_.SetIteration(POI_.Iteration() + 1);

		//-! Fill warpped image into Subset T
		fSubAveT = 0;
		fSubNorT = 0;
		for (int l = 0; l < iSubsetD; l++)
		{
			for (int m = 0; m < iSubsetH; m++)
			{
				float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
				for (int n = 0; n < iSubsetW; n++)
				{
					//-! Calculate the location of warped subvolume T
					fWarpX =
						POI_.X() + 1 + m_fWarp[0][0] * (n - iSubsetX) + m_fWarp[0][1] * (m - iSubsetY) + m_fWarp[0][2] * (l - iSubsetZ) + m_fWarp[0][3];
					fWarpY =
						POI_.Y() + 1 + m_fWarp[1][0] * (n - iSubsetX) + m_fWarp[1][1] * (m - iSubsetY) + m_fWarp[1][2] * (l - iSubsetZ) + m_fWarp[1][3];
					fWarpZ =
						POI_.Z() + 1 + m_fWarp[2][0] * (n - iSubsetX) + m_fWarp[2][1] * (m - iSubsetY) + m_fWarp[2][2] * (l - iSubsetZ) + m_fWarp[2][3];
					iTempX = int(fWarpX);
					iTempY = int(fWarpY);
					iTempZ = int(fWarpZ);

					m_fTempX = fWarpX - iTempX;
					m_fTempY = fWarpY - iTempY;
					m_fTempZ = fWarpZ - iTempZ;
					if ((iTempX >= 1) && (iTempY >= 1) && (iTempZ >= 1) &&
						(iTempX <= (iOriginVolWidth - 3 )) && (iTempY <= (iOriginVolHeight - 3 )) && (iTempZ <= (iOriginVolDepth - 3 )))
					{
						//-! In most case, it is sub-pixel location, estimate the gary intensity using interpolation
						m_fSubsetT_id_lm[n] = 0;

						w_x[0] = w0_c(m_fTempX);
						w_x[1] = w1_c(m_fTempX);
						w_x[2] = w2_c(m_fTempX);
						w_x[3] = w3_c(m_fTempX);

						w_y[0] = w0_c(m_fTempY);
						w_y[1] = w1_c(m_fTempY);
						w_y[2] = w2_c(m_fTempY);
						w_y[3] = w3_c(m_fTempY);

						w_z[0] = w0_c(m_fTempZ);
						w_z[1] = w1_c(m_fTempZ);
						w_z[2] = w2_c(m_fTempZ);
						w_z[3] = w3_c(m_fTempZ);

						//z
						for (int j = 0; j<4; ++j) {
							//y
							float **m_fTGBspline_j = m_fTGBspline[iTempZ + 2 - 1 + j];
							for (int i = 0; i<4; ++i) {
								float *m_fTGBspline_ji = m_fTGBspline_j[iTempY + 2 - 1 + i];
								/*sum_x[i] =
									w_x[0] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 - 1]
									+ w_x[1] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1]
									+ w_x[2] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 1]
									+ w_x[3] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 2];*/
								sum_x[i] =
									  w_x[0] * m_fTGBspline_ji[iTempX + 2 - 1]
									+ w_x[1] * m_fTGBspline_ji[iTempX + 2	 ]
									+ w_x[2] * m_fTGBspline_ji[iTempX + 2 + 1]
									+ w_x[3] * m_fTGBspline_ji[iTempX + 2 + 2];
							}
							sum_y[j] =
								w_y[0] * sum_x[0]
								+ w_y[1] * sum_x[1]
								+ w_y[2] * sum_x[2]
								+ w_y[3] * sum_x[3];
						}

						m_fSubsetT_id_lm[n] =
							w_z[0] * sum_y[0]
							+ w_z[1] * sum_y[1]
							+ w_z[2] * sum_y[2]
							+ w_z[3] * sum_y[3];

#ifdef _JR_DEBUG
						if (l==0 && m==16 && n ==0) {
							printf("id:%d; PRef:(%d,%d,%d); Ptar:(%f,%f,%f); val:%f; iTempX:(%d,%d,%d); inerp:(%f)\n", ((l*iSubsetH + m)*iSubsetW + n), n, m, l, fWarpX, fWarpY, fWarpZ, m_fSubsetT_id_lm[n], iTempX, iTempY, iTempZ, m_fTGBspline[iTempZ + 2][iTempY + 2][iTempX + 2]);
						}
#endif

#ifdef _JR_DEBUG
						if (((l*iSubsetH + m)*iSubsetW + n) == 0) {
							printf("id:%d; PRef:(%d,%d,%d); Ptar:(%f,%f,%f); val:%f; iTempX:(%d,%d,%d); inerp:(%f)\n", ((l*iSubsetH + m)*iSubsetW + n), n, m, l, fWarpX, fWarpY, fWarpZ, m_fSubsetT_id_lm[n], iTempX, iTempY, iTempZ, m_fTGBspline[iTempZ + 2][iTempY + 2][iTempX + 2]);
						}
#endif

						fSubAveT += m_fSubsetT_id_lm[n];
					}
					else
					{
						//-! if the loacation of the warped subvolume T is out of the ROI, stop iteration and set p as the current value
						POI_.SetOutofROI(1);
						POI_.SetProcessed(-2);
						return;
					}
				}
			}
		}
		fSubAveT /= fSubsetSize;
		//-! Check if Subset T is a all dark subvolume
		if (fSubAveT == 0)
		{
			POI_.SetDarkSubset(4);
			POI_.SetProcessed(-1);
			return;
		}
		for (int l = 0; l < iSubsetD; l++)
		{
			for (int m = 0; m < iSubsetH; m++)
			{
				for (int n = 0; n < iSubsetW; n++)
				{
					m_fSubsetT_id[l][m][n] = m_fSubsetT_id[l][m][n] - fSubAveT;	// T_i - T_m
					fSubNorT += pow(m_fSubsetT_id[l][m][n], 2);				// Sigma(T_i - T_m)^2
				}
			}
		}
		fSubNorT = sqrt(fSubNorT);	// sqrt (Sigma(T_i - T_m)^2)
		if (fSubNorT == 0)
		{
			POI_.SetDarkSubset(5);	// set flag
			POI_.SetProcessed(-1);
			return;
		}


#ifdef _JR_DEBUG
		printf("iteration%d, normT:%f, normR/normT:%f\n", POI_.Iteration()-1, fSubNorT, fSubNorR / fSubNorT);
#endif

		//-! Compute the error image
		for (int k = 0; k < 12; k++)
		{
			m_fNumerator[k] = 0;
		}
		fZNCC = 0;
		for (int l = 0; l < iSubsetD; l++)
		{
			for (int m = 0; m < iSubsetH; m++)
			{
				float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
				float *m_fSubsetR_id_lm = m_fSubsetR_id[l][m];
				for (int n = 0; n < iSubsetW; n++)
				{
					//fErrors = (fSubNorR / fSubNorT) * m_fSubsetT_id[l][m][n] - m_fSubsetR_id[l][m][n];
					fErrors = (fSubNorR / fSubNorT) * m_fSubsetT_id_lm[n] - m_fSubsetR_id_lm[n];
					fZNCC += m_fSubsetT_id[l][m][n] * m_fSubsetR_id[l][m][n];
					//-! Compute the numerator
					float *m_fRDescent_id_lmn = m_fRDescent_id[l][m][n];
					for (int k = 0; k < 12; k++)
					{
						//m_fNumerator[k] += (m_fRDescent_id[l][m][n][k] * fErrors);
						m_fNumerator[k] += (m_fRDescent_id_lmn[k] * fErrors);
					}
				}
			}
		}
		fZNCC = fZNCC / (fSubNorR*fSubNorT);
		POI_.SetZNCC(fZNCC);
		//-! Compute DeltaP
		for (int k = 0; k < 12; k++)
		{
			DP[k] = 0;
			for (int n = 0; n < 12; n++)
			{
				DP[k] += (m_fInvHessian[k][n] * m_fNumerator[n]);
			}
		}
		POI_.SetDP(DP);
		fDU = POI_.DP()[0];
		fDUx = POI_.DP()[1];
		fDUy = POI_.DP()[2];
		fDUz = POI_.DP()[3];
		fDV = POI_.DP()[4];
		fDVx = POI_.DP()[5];
		fDVy = POI_.DP()[6];
		fDVz = POI_.DP()[7];
		fDW = POI_.DP()[8];
		fDWx = POI_.DP()[9];
		fDWy = POI_.DP()[10];
		fDWz = POI_.DP()[11];

		//-! Update the warp
		//-! m_fTemp: store the denominator of the Inverse W
		m_fTemp =
			((fDWz + 1)*fDVy + (-fDWy*fDVz + (fDWz + 1)))*fDUx +
			(((-fDWz - 1)*fDVx + fDWx*fDVz)*fDUy + ((fDWy*fDVx + (-fDWx*fDVy - fDWx))*fDUz +
			((fDWz + 1)*fDVy + (-fDWy*fDVz + (fDWz + 1)))));
		if (m_fTemp == 0)
		{
			POI_.SetInvertibleHessian(2);
			POI_.SetProcessed(-1);
			return;
		}

		//-! W(P) <- W(P) o W(DP)^-1
		f00 = (fDWz + 1)*fDVy + (-fDWy*fDVz + (fDWz + 1));
		f10 = (-fDWz - 1)*fDVx + fDWx*fDVz;
		f20 = fDWy*fDVx + (-fDWx*fDVy - fDWx);
		f01 = (-fDWz - 1)*fDUy + fDWy*fDUz;
		f11 = (fDWz + 1)*fDUx + (-fDWx*fDUz + (fDWz + 1));
		f21 = -fDWy*fDUx + (fDWx*fDUy - fDWy);
		f02 = fDVz*fDUy + (-fDVy - 1)*fDUz;
		f12 = -fDVz*fDUx + (fDVx*fDUz - fDVz);
		f22 = (fDVy + 1)*fDUx + (-fDVx*fDUy + (fDVy + 1));
		f03 = (-fDW*fDVz + (fDWz + 1)*fDV)*fDUy + ((fDW*fDVy + (-fDWy*fDV + fDW))*fDUz + ((-fDWz - 1)*fDVy + (fDWy*fDVz + (-fDWz - 1)))*fDU);
		f13 = (fDW*fDVz + (-fDWz - 1)*fDV)*fDUx + ((-fDW*fDVx + fDWx*fDV)*fDUz + (((fDWz + 1)*fDVx - fDWx*fDVz)*fDU + (fDW*fDVz + (-fDWz - 1)*fDV)));
		f23 = (-fDW*fDVy + (fDWy*fDV - fDW))*fDUx + ((fDW*fDVx - fDWx*fDV)*fDUy + ((-fDWy*fDVx + (fDWx*fDVy + fDWx))*fDU + (-fDW*fDVy + (fDWy*fDV - fDW))));
		m_fWarp[0][0] = ((1 + fUx)*f00 + fUy*f10 + fUz*f20) / m_fTemp;	m_fWarp[0][1] = ((1 + fUx)*f01 + fUy*f11 + fUz*f21) / m_fTemp; 	m_fWarp[0][2] = ((1 + fUx)*f02 + fUy*f12 + fUz*f22) / m_fTemp; 	m_fWarp[0][3] = ((1 + fUx)*f03 + fUy*f13 + fUz*f23) / m_fTemp + fU;
		m_fWarp[1][0] = (fVx*f00 + (1 + fVy)*f10 + fVz*f20) / m_fTemp;	m_fWarp[1][1] = (fVx*f01 + (1 + fVy)*f11 + fVz*f21) / m_fTemp;	m_fWarp[1][2] = (fVx*f02 + (1 + fVy)*f12 + fVz*f22) / m_fTemp;	m_fWarp[1][3] = (fVx*f03 + (1 + fVy)*f13 + fVz*f23) / m_fTemp + fV;
		m_fWarp[2][0] = (fWx*f00 + fWy*f10 + (1 + fWz)*f20) / m_fTemp;	m_fWarp[2][1] = (fWx*f01 + fWy*f11 + (1 + fWz)*f21) / m_fTemp;	m_fWarp[2][2] = (fWx*f02 + fWy*f12 + (1 + fWz)*f22) / m_fTemp;	m_fWarp[2][3] = (fWx*f03 + fWy*f13 + (1 + fWz)*f23) / m_fTemp + fW;
		m_fWarp[3][0] = 0;	m_fWarp[3][1] = 0;	m_fWarp[3][2] = 0;	m_fWarp[3][3] = 1;

		//-! Update P
		P[0] = m_fWarp[0][3];	P[1] = m_fWarp[0][0] - 1;	P[2] = m_fWarp[0][1];		P[3] = m_fWarp[0][2];
		P[4] = m_fWarp[1][3];	P[5] = m_fWarp[1][0];		P[6] = m_fWarp[1][1] - 1;	P[7] = m_fWarp[1][2];
		P[8] = m_fWarp[2][3];	P[9] = m_fWarp[2][0];		P[10] = m_fWarp[2][1];		P[11] = m_fWarp[2][2] - 1;
		POI_.SetP(P);
		fU = POI_.P()[0];
		fUx = POI_.P()[1];
		fUy = POI_.P()[2];
		fUz = POI_.P()[3];
		fV = POI_.P()[4];
		fVx = POI_.P()[5];
		fVy = POI_.P()[6];
		fVz = POI_.P()[7];
		fW = POI_.P()[8];
		fWx = POI_.P()[9];
		fWy = POI_.P()[10];
		fWz = POI_.P()[11];

#ifdef _JR_DEBUG
		printf("iteration: %d, deltaP:%f\n", POI_.Iteration() - 1, sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)));
		printf("\t\tmatrix:%f %f %f %f, %f %f %f %f, %f %f %f %f \n",
			P[0], P[1], P[2], P[3],
			P[4], P[5], P[6], P[7],
			P[8], P[9], P[10], P[11]);
#endif
	}


	if (sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)) <= fDeltaP)
		POI_.SetConverge(1);

	POI_.SetProcessed(1);
	return;
}

void CPaDVC::Strategy3_simple_transfer() {

	double t_Start_S3 = omp_get_wtime();

	deque<int> deque_idx_1, deque_idx_2;

	default_random_engine ran_en(time(NULL));
	uniform_int_distribution<int> ran_distri;

	{
		//assign shuffled index to 
		vector<int> tmp_vi;
		for (int i = 0; i < m_POI_global.size(); ++i) {
			if (m_POI_global[i].GetEmpty() == 1)
				tmp_vi.push_back(i);
		};
		shuffle(tmp_vi.begin(), tmp_vi.end(), ran_en);
		deque_idx_1.assign(tmp_vi.begin(), tmp_vi.end());
	}

	if (deque_idx_1.size() == m_POI_global.size()) {
		cout << "No available poi that has been processed" << endl;
		return;
	}

	vector<int> toBe;
	size_t batchSize = (thread_num - 1) * 1;
	thread *t_process = nullptr;
	while (deque_idx_1.size()) {
		while (deque_idx_1.size() && toBe.size() < batchSize) {
			int currIdx = deque_idx_1.front();
			deque_idx_1.pop_front();
#ifdef  _DEBUG
			if (m_POI_global[currIdx].G_X() == 620 && m_POI_global[currIdx].G_Y() == 260 && m_POI_global[currIdx].G_Z() == 59) {
				cout << "catched" << endl;
			}
#endif 
			int neighborId[6];
			//int processedNum = Collect_neighborPOI(m_POI_global[currIdx], neighborId);
			int processedNum = Collect_neighborBestPOI(m_POI_global[currIdx], neighborId);
			if (processedNum == 0) {
				//push back to idx2, avoid looping infinitely
				deque_idx_2.push_back(currIdx);
				continue;
			}
			//int chosen = neighborId[(ran_distri(ran_en)) % processedNum];
			int chosen = neighborId[0];
			m_POI_global[currIdx].SetP0(Calculate_P0_full(m_POI_global[chosen], m_POI_global[currIdx]));
			m_POI_global[currIdx].SetStrategy(Vector_Transfer);
			m_POI_global[currIdx].search_radius = chosen;
			toBe.push_back(currIdx);
		}

		if (t_process) {
			t_process->join();
			delete t_process;
		}
		if (toBe.size()) {
			t_process = new thread(&CPaDVC::BatchICGN, this, toBe, thread_num - 1);
			//cout << "Start processing size of:" << toBe.size() << endl; 
		}
		else
			t_process = nullptr;

		// make it back to deque1
		for (int i = 0; i < deque_idx_2.size(); ++i) {
			int idx = deque_idx_2.front();
			deque_idx_2.pop_front();
			deque_idx_1.push_back(idx);
		}
		toBe.clear();
		//cout << "remaining:" << deque_idx_1.size()  << endl;
	}

	if (t_process) {
		t_process->join();
		delete t_process;
	}
}

void CPaDVC::FFTCC_MODIFY(CPOI &POI_, int iID_, float znccThres, int iSubsetX, int iSubsetY, int iSubsetZ)
{

	if (POI_.G_X() - iSubsetX<0 || POI_.G_X() + iSubsetX >= m_iOriginVolWidth ||
		POI_.G_Y() - iSubsetY<0 || POI_.G_Y() + iSubsetY >= m_iOriginVolHeight ||
		POI_.G_Z() - iSubsetZ<0 || POI_.G_Z() + iSubsetZ >= m_iOriginVolDepth) {
		return;
	}

	int iFFTSubW = iSubsetX * 2;
	int iFFTSubH = iSubsetY * 2;
	int iFFTSubD = iSubsetZ * 2;
	int iFFTSize = iFFTSubW * iFFTSubH * iFFTSubD;
	int iFFTFreqSize = iFFTSubW*iFFTSubH*(iFFTSubD / 2 + 1);

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

	if (fTempZNCC>znccThres) {
		vector<float>tempP0(12, 0);
		tempP0[0] = float(m_iU);
		tempP0[4] = float(m_iV);
		tempP0[8] = float(m_iW);
		POI_.SetStrategy(FFTCC);
		POI_.SetEmpty(0);
		POI_.SetZNCC(fTempZNCC);
		POI_.SetP0(tempP0);
		cout << "Got FFTCC:" << POI_.G_X() << "," << POI_.G_Y() << "," << POI_.G_Z() << "," << endl;
	}
}

void CPaDVC::FFTCC_AUTO_EXPAND(CPOI &POI_)
{
}

 //-! 4.Output
void CPaDVC::SaveResult2Text_global_d(const string qstrOutputPath_,
	SIFT_PROCESS SIFT_TIME, GuessMethod initMethod,
	const string  c_ref_, const string  c_tar_,
	const string  f_ref_, const string  f_tar_) {

	//ofstream oFile;
	ofstream oFile;
	oFile.open(qstrOutputPath_, ios::out | ios::trunc);

	oFile << "Image size" << "," << "Subvol Size" << "," << "Num of threads" << "," << "Interpolation Method" << endl
		<< m_iOriginVolWidth << " X " << m_iOriginVolHeight << " X " << m_iOriginVolDepth << ","
		<< m_iSubsetX * 2 + 1 << " X " << m_iSubsetY * 2 + 1 << " X " << m_iSubsetZ * 2 + 1 << "," << thread_num << ",";

	if (this->m_eInterpolation == BSPLINE)
		oFile << "BSPLINE" << endl;
	else if (this->m_eInterpolation == TRICUBIC)
		oFile << "TRICUBIC" << endl;
	
	static std::map<GuessMethod, std::string> guessNameMap = {
		{ PiSIFT, std::string("PiSIFT") },
		{ IOPreset, std::string("IOPreset") },
		{ FFTCCGuess, std::string("FFT-CC") },
	};
	oFile << "GuessMethod:" << guessNameMap[initMethod] << std::endl;

	if (initMethod == PiSIFT) {
		oFile << "------PiSIFT guess parameters:------" << std::endl;
		oFile << "Minimum neighor num:" << m_iMinNeighbor << "\n"
			<< "Ransac error epsilon:" << ransacCompute.ransacErrorEpsilon << "\n"
			<< "Ransac iter num:" << ransacCompute.ransacMaxIterNum << "\n"
			<< "expand serach radius ratio:" << m_iExpandRatio << "\n" 
			<< "Number of matches:" << Ref_point.size() << std::endl;
		oFile << "------------------------------------" << std::endl;
	}

	oFile << "----------ICGN parameters:----------" << std::endl;
	oFile << "deltaP:" << m_dDeltaP << "\n"
		  << "IC-GN max iter:" << m_iMaxIterationNum << std::endl;
	oFile << "------------------------------------" << std::endl;

	oFile << "Sift Part Time:" << endl;
	oFile << SIFT_TIME << endl;

	oFile << "Time:" << endl;
	oFile << "Preparation, Precomputation, Build KD-Tree, AFfine Calculation, FFT-CC, ICGN Calculation, Transfer, Total" << endl;
	oFile << m_Timer.dPreparationTime << ","
		<< m_Timer.dPrecomputationTime << ","
		<< m_Timer.dBuildKDTreeTime << ","
		<< m_Timer.dLocalAffineTime << ","
		<< m_Timer.dFFTCcTime << ","
		<< m_Timer.dICGNTime << ","
		<< m_Timer.dTransferStrategyTime << ","
		<< m_Timer.dConsumedTime << endl;


	oFile << "Calculation Results: " << endl << endl;

	oFile << "posX" << "," << "posY" << "," << "posZ" << "," << "ZNCC Value" << ","
		<< "U-displacement" << "," << "V-displacement" << "," << "W-displacement"
		<< "," << "U0" << "," << "V0" << "," << "W0"
		<< "," << "Ux" << "," << "Uy" << "," << "Uz"
		<< "," << "Vx" << "," << "Vy" << "," << "Vz"
		<< "," << "Wx" << "," << "Wy" << "," << "Wz"
		<< "," << "Number of Iterations"
		<< "," << "Range Enough"
		<< "," << "Candidate Num"
		<< "," << "Final Ransac Num"
		//debug for KP
		<< "," << "Edge Flag"
		<< "," << "Out_of_ROI_flag"
		<< "," << "Grayvalue of POI"
		<< "," << "Mean distance" 
		<< "," << "Strategy"
		<< "," << "Converge"
		<< "," << "searchRadius" << endl;

	cout << m_POI_global.size() << endl;


	for (int i = 0; i < m_POI_global.size(); i++) {

		auto P = m_POI_global[i].P();
		auto P0 = m_POI_global[i].P0();

		oFile
			<< m_POI_global[i].G_X() << ","
			<< m_POI_global[i].G_Y() << ","
			<< m_POI_global[i].G_Z() << ","
			<< m_POI_global[i].ZNCC() << ","
			<< P[0] << ","
			<< P[4] << ","
			<< P[8] << ","
			//
			<< P0[0] << ","
			<< P0[4] << ","
			<< P0[8] << ","
			//
			<< P[1] << ","
			<< P[2] << ","
			<< P[3] << ","
			//
			<< P[5] << ","
			<< P[6] << ","
			<< P[7] << ","
			//
			<< P[9] << ","
			<< P[10] << ","
			<< P[11] << ","
			//
			<< m_POI_global[i].Iteration() << ","
			<< m_POI_global[i].GetRangeGood() << ","
			<< m_POI_global[i].num_candidate << ","
			<< m_POI_global[i].num_final << ","
			<< m_POI_global[i].GetEdge() << ","
			<< m_POI_global[i].isOutofROI();
		//float AffineMatrix[12];
		//m_POI_global[i].GetAffineMatrix(AffineMatrix);

		/*
		for (int l = 0; l < 12; ++l)
		oFile << "," << m_POI_global[i].m_fAffine[l];
		*/
		/*
		oFile << "," << m_fVolT[ELT(m_iOriginVolHeight, m_iOriginVolWidth,
		k*m_iGridSpaceX + m_iOriginMarginX + m_iSubsetX,	// x
		j*m_iGridSpaceY + m_iOriginMarginY + m_iSubsetY,	// y
		i*m_iGridSpaceZ + m_iOriginMarginZ + m_iSubsetZ)];	// z;
		*/
		oFile << "," << m_POI_global[i].GrayValue
			<< "," << m_POI_global[i].Dist
			<< "," << m_POI_global[i].strategy
			<< "," << m_POI_global[i].GetConverge()
			<< "," << m_POI_global[i].search_radius;

		oFile << endl;

	}
	oFile.close();


	//debug 
	ofstream oFile_cref;
	ofstream oFile_ctar;
	ofstream oFile_fref;
	ofstream oFile_ftar;
	if (c_ref_.size() > 1) {
		oFile_cref.open(c_ref_, ios::out | ios::trunc);
	}
	else {
		oFile_cref.setstate(std::ios_base::badbit);
	}

	if (c_tar_.size() > 1) {
		oFile_ctar.open(c_tar_, ios::out | ios::trunc);
	}
	else {
		oFile_ctar.setstate(std::ios_base::badbit);
	}

	if (f_ref_.size() > 1) {
		oFile_fref.open(f_ref_, ios::out | ios::trunc);
	}
	else {
		oFile_fref.setstate(std::ios_base::badbit);
	}

	if (f_tar_.size() > 1) {
		oFile_ftar.open(f_tar_, ios::out | ios::trunc);
	}
	else {
		oFile_ftar.setstate(std::ios_base::badbit);
	}


	for (int i = 0; i < m_POI_global.size(); i++) {

		oFile_cref << m_POI_global[i].num_candidate;
		for (int index_candidate = 0; index_candidate < m_POI_global[i].c_ref.size(); ++index_candidate) {
			oFile_cref
				<< "," << m_POI_global[i].c_ref[index_candidate].x
				<< "," << m_POI_global[i].c_ref[index_candidate].y
				<< "," << m_POI_global[i].c_ref[index_candidate].z;
		}

		sort(m_POI_global[i].s_idx.begin(), m_POI_global[i].s_idx.end());
		for (auto j : m_POI_global[i].s_idx) 
			oFile_cref << j << ",";
		oFile_cref << endl;

		oFile_ctar << m_POI_global[i].num_candidate;
		for (int index_candidate = 0; index_candidate < m_POI_global[i].c_tar.size(); ++index_candidate) {
		oFile_ctar
			<< "," << m_POI_global[i].c_tar[index_candidate].x
			<< "," << m_POI_global[i].c_tar[index_candidate].y
			<< "," << m_POI_global[i].c_tar[index_candidate].z;
		}
		oFile_ctar << endl;

		oFile_fref << m_POI_global[i].num_final;
		for (int index_final = 0; index_final < m_POI_global[i].f_ref.size(); ++index_final) {
			oFile_fref
				<< "," << m_POI_global[i].f_ref[index_final].x
				<< "," << m_POI_global[i].f_ref[index_final].y
				<< "," << m_POI_global[i].f_ref[index_final].z;

		}
		oFile_ftar << endl;

		oFile_ftar << m_POI_global[i].num_final;
		for (int index_final = 0; index_final < m_POI_global[i].f_tar.size(); ++index_final) {

			oFile_ftar
				<< "," << m_POI_global[i].f_tar[index_final].x
				<< "," << m_POI_global[i].f_tar[index_final].y
				<< "," << m_POI_global[i].f_tar[index_final].z;
		}
		oFile_fref << endl;

	}

	if (c_ref_.size()>1)
		oFile_cref.close();
	if (c_tar_.size()>1)
		oFile_ctar.close();
	if (f_ref_.size()>1)
		oFile_fref.close();
	if (f_tar_.size()>1)
		oFile_ftar.close();

}
