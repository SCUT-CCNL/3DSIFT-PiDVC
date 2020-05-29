#include "../Include/mulDVC.h"

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

#include <fftw3.h>
#include "../Include/kdTreeUtil.h"
#include "../Include/MemManager.h"
#include "../Include/matrixIO3D.h"
#include "../Include/FitFormula.h"
#include "../Include/Spline.h"
#include <3DSIFT\Include\Util\readNii.h>

using namespace std;
using namespace CPUSIFT;

//#define _JR_DEBUG

#ifdef _DEBUG
	int debugP[3] = { 170,160, 8 };
#endif

/*
	tools func used in calculation
*/
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

	omp_set_num_threads(thread_num);

	//set subset
	fftccGuessCompute.setSubsetRadius(m_iSubsetX, m_iSubsetY, m_iSubsetZ);
	siftGuessCompute.setSubsetRadius(m_iSubsetX, m_iSubsetY, m_iSubsetZ);
	icgnCompute.setSubsetRadius(m_iSubsetX, m_iSubsetY, m_iSubsetZ);

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
		
		fftccGuessCompute.free();
		siftGuessCompute.free();
		icgnCompute.free();

		Destroy();
	}
	else
	{
		Destroy();
	}
}

void CPaDVC::ReadVol(const string chFile1, const string chFile2) {

	//-! 1. Load and compare two voxels
    m_fVolR = readNiiFile(chFile1.c_str(), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
    
    int m, n, p;
    m_fVolT = readNiiFile(chFile2.c_str(), m, n, p);

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
	siftGuessCompute.setParameters(minNeighNum, 1.2 , errorEpsilon, maxIter);
}

void CPaDVC::SetICGNParam(const float deltaP, const int maxIter) {
	icgnCompute.setParameters(deltaP, maxIter);
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
	//-! 1. Prepare
	double t1 = omp_get_wtime();
	Precomputation_Prepare_global();
	siftGuessCompute.init(thread_num, Ref_point, Tar_point);
	icgnCompute.init(thread_num, m_fVolR, m_fVolT, m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth,
		m_fRx, m_fRy, m_fRz);
	double t_PreICGN_end = omp_get_wtime();
	cout << "----!1.Finish Preparation in " << t_PreICGN_end - t1 << "s" << endl;

	//---------------------Precompute---------------------------------
	//-! 2. Precompute
	//BSPLINE PREFILTER
	double t_Prefilter_start = omp_get_wtime();
	Prefilter(m_fVolT, (m_fTGBspline[0][0]), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	TransSplineArray();
    icgnCompute.setBSplineTable(m_fTGBspline);
	PrecomputeGradients_mul_global();
	double t_Precompute_End = omp_get_wtime();
	cout << "----!2-1.Finish Precomputation of Images in " << t_Precompute_End - t_Prefilter_start << "s" << endl;
	//Using SIFT and kdtree to finish
	siftGuessCompute.preCompute();
	double t_kdtree_end = omp_get_wtime();
	cout << "----!2-2.Finish Building KD-tree of SIFT Keypoints in " << t_kdtree_end - t_Precompute_End << "s" << endl;


	//------------------------Initial guess------------------------------
	//-! 3. SIFT-Aided
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); i++)
	{
		siftGuessCompute.compute(m_POI_global[i]);
	}
#pragma omp barrier
	double t_aff1 = omp_get_wtime();
	cout << "----!3-1.Finish Affine Initial Guess in " << t_aff1 - t_kdtree_end << "s" << endl;

	//--------------------High-accuracy Registration--------------------
	//-! 4. ICGN
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); ++i) {
		if (m_POI_global[i].GetEmpty() == 0) {
			icgnCompute.compute(m_POI_global[i]);
		}
		if (i % int(m_POI_global.size() / 10) == 0) {
			cout << "Processing POI " << i / int(m_POI_global.size() / 10) << "0%" << endl;
		}
	}
	double t_ICGN_End = omp_get_wtime();
	cout << "----!4.Finish ICGN in " << t_ICGN_End - t_aff1 << "s" << endl;

	//-------------------- Back-up Strategy for SIFT-aided--------------------
	//-! 5. Strategy 3
	Strategy3_simple_transfer();
	double t_Strategy3_End = omp_get_wtime();
	cout << "----!5.Finish Strategy3 in " << t_Strategy3_End - t_ICGN_End << "s" << endl;

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
	//-! 1. Prepare
	double t1 = omp_get_wtime();
	Precomputation_Prepare_global();
	fftccGuessCompute.init(thread_num, m_fVolR, m_fVolT, m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	icgnCompute.init(thread_num, m_fVolR, m_fVolT, m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth, 
		m_fRx, m_fRy, m_fRz);
	double t_PreICGN_end = omp_get_wtime();
	cout << "----!1.Finish Preparation in " << t_PreICGN_end - t1 << "s" << endl;

	//---------------------Precompute---------------------------------
	//-! 2. Precompute
	//BSPLINE PREFILTER
	double t_Prefilter_start = omp_get_wtime();
	CMemManager<float>::hCreatePtr(m_fTGBspline, m_iOriginVolDepth, m_iOriginVolHeight, m_iOriginVolWidth);
	Prefilter(m_fVolT, (&m_fTGBspline[0][0][0]), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	TransSplineArray();
    icgnCompute.setBSplineTable(m_fTGBspline);
	double t_Prefilter_end = omp_get_wtime();
	cout << "----!2.Finish Prefilter in " << t_Prefilter_end - t_Prefilter_start << "s" << endl;
	PrecomputeGradients_mul_global();
	double t_Precompute_End = omp_get_wtime();
	cout << "----!2.Finish Precomputation of Gradients in " << t_Precompute_End - t_Prefilter_end << "s" << endl;

	//------------------------Initial guess------------------------------
	//-! 3. FFT-CC
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); i++)
	{
		fftccGuessCompute.compute(m_POI_global[i]);
	}
#pragma omp barrier
	double t_FFTCc_End = omp_get_wtime();
	cout << "----!3.Finish FFT-CC Initial Guess in " << t_FFTCc_End - t_Precompute_End << "s" << endl;

	//--------------------High-accuracy Registration--------------------
	//-! 4. ICGN
	int total_POI_num = m_POI_global.size();
	int tenPercent_num = total_POI_num / 10;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); ++i) {
		icgnCompute.compute(m_POI_global[i]);
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
	icgnCompute.init(thread_num, m_fVolR, m_fVolT, m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth,
		m_fRx, m_fRy, m_fRz);
	double t_PreICGN_end = omp_get_wtime();
	cout << "----!1.Finish Preparation in " << t_PreICGN_end - t1 << "s" << endl;

	//BSPLINE PREFILTER
	double t_Prefilter_start = omp_get_wtime();
	CMemManager<float>::hCreatePtr(m_fTGBspline, m_iOriginVolDepth, m_iOriginVolHeight, m_iOriginVolWidth);
	Prefilter(m_fVolT, (&m_fTGBspline[0][0][0]), m_iOriginVolWidth, m_iOriginVolHeight, m_iOriginVolDepth);
	TransSplineArray();
    icgnCompute.setBSplineTable(m_fTGBspline);
	double t_Prefilter_end = omp_get_wtime();
	cout << "----!2.Finish Prefilter in " << t_Prefilter_end - t_Prefilter_start << "s" << endl;
	PrecomputeGradients_mul_global();
	double t_Precompute_End = omp_get_wtime();
	cout << "----!2.Finish Precomputation in " << t_Precompute_End - t_Prefilter_end << "s" << endl;

	//ICGN
	const int total_POI_num = m_POI_global.size();
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for (int i = 0; i < m_POI_global.size(); ++i) {
		icgnCompute.compute(m_POI_global[i]);
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

 //-! 3.Compute Utils
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
		icgnCompute.compute(m_POI_global[poi_id]);
	}
	return;
}

 //-! 3-2.Computation funs
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
//#ifdef  _DEBUG
//			if (m_POI_global[currIdx].G_X() == 620 && m_POI_global[currIdx].G_Y() == 260 && m_POI_global[currIdx].G_Z() == 59) {
//				cout << "catched" << endl;
//			}
//#endif 
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

 //-! 4.Output
void CPaDVC::SaveResult2Text_global_d(const string qstrOutputPath_,
	SIFT_PROCESS SIFT_TIME, GuessMethod initMethod) {

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
		oFile << "Minimum neighor num:" << siftGuessCompute.getMinNeighborNum()<< "\n"
			<< "Ransac error epsilon:" << siftGuessCompute.getRansacEpsilon()<< "\n"
			<< "Ransac iter num:" << siftGuessCompute.getRansacIter() << "\n"
			<< "expand serach radius ratio:" << siftGuessCompute.getExpandRatio() << "\n"
			<< "Number of matches:" << Ref_point.size() << std::endl;
		oFile << "------------------------------------" << std::endl;
	}

	oFile << "----------ICGN parameters:----------" << std::endl;
	oFile << "deltaP:" << icgnCompute.getDeltaP() << "\n"
		  << "IC-GN max iter:" << icgnCompute.getMaxIter() << std::endl;
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

	oFile << "posX" << "," << "posY" << "," << "posZ" << "," << "ZNCC_Value" << ","
		<< "U-displacement" << "," << "V-displacement" << "," << "W-displacement"
		<< "," << "U0" << "," << "V0" << "," << "W0"
		<< "," << "Ux" << "," << "Uy" << "," << "Uz"
		<< "," << "Vx" << "," << "Vy" << "," << "Vz"
		<< "," << "Wx" << "," << "Wy" << "," << "Wz"

		<< "," << "IterationNumber"
        << "," << "OutROIflag"
        << "," << "Converge"

        << "," << "Strategy"
		<< "," << "CandidateNum"
		<< "," << "FinalRansacNum"
        /*
        			<< m_POI_global[i].Iteration() << ","
            << m_POI_global[i].isOutofROI() << ","
            << m_POI_global[i].GetConverge() << ","
            << m_POI_global[i].strategy << ","
            << m_POI_global[i].num_candidate << ","
            << m_POI_global[i].num_final;*/
        << endl;

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
            << m_POI_global[i].isOutofROI() << ","
            << m_POI_global[i].GetConverge() << ","
            << m_POI_global[i].strategy << ","
            << m_POI_global[i].num_candidate << ","
            << m_POI_global[i].num_final;

		oFile << endl;

	}
	oFile.close();

}
