#ifndef __PIDVC_CPOI_H__
#define __PIDVC_CPOI_H__

//-! Class for an individual POI

//#include "SIFT3D.h"
#include <array>
#include <vector>
#include <string>
#include <3DSIFT/Include/cSIFT3D.h>

struct int_3d {
	int x;
	int y;
	int z;
	int_3d() {
		x = y = z = 0;
	}
	int_3d(int x_, int y_, int z_): x(x_), y(y_), z(z_) {}
};

enum InitStrategy {
	Unprocessed = 0,
	Search_Subset = 1,
	Search_Radius = 2,
	Search_Expand_Radius = 3,
	Kp_Vector_Transfer = 7,
	FFTCC = 10,
	Vector_Transfer = 30,
	Preset = 100
};

class CPOI
{
public:
	CPOI()
		:x(0), y(0), z(0), m_fZNCC(0), m_iProcessed(0), m_dFFTCCTime(0), m_dICGNTime(0),
		m_iIteration(0), m_iDarkSubset(0), m_iOutofROI(0), m_iInvertibleHessian(0),
		m_vP(12, 0), m_vDP(12, 0), m_vP0(12, 0), 
		m_iRangeGood(0), m_iEdge(0), m_iEmpty(0), m_iConverge(0), strategy(Unprocessed)
		//, m_fNeighbourZNCC(0)
	{
		//-! Default Constructor
		for (int i = 0; i < 6; ++i)
			neighbor[i] = -1;
	}
	
	CPOI(int x_, int y_, int z_)
		:x(x_), y(y_), z(z_), m_fZNCC(0), m_iProcessed(0), m_dFFTCCTime(0), m_dICGNTime(0),
		m_iIteration(0), m_iDarkSubset(0), m_iOutofROI(0), m_iInvertibleHessian(0),
		m_vP(12, 0), m_vDP(12, 0), m_vP0(12, 0), m_iRangeGood(0), m_iEdge(0), m_iEmpty(0), m_iConverge(0), strategy(Unprocessed)
		//, m_fNeighbourZNCC(0)
	{
		for (int i = 0; i < 6; ++i)
			neighbor[i] = -1;
	}

	~CPOI()
	{}

	//-! Getters
	//-! Get x,y,z coordinates of a POI
	inline int X() { return x; }
	inline int Y() { return y; }
	inline int Z() { return z; }
	//-! Get Global x,y,z coordinates of a POI
	inline int G_X() { return G_x; }
	inline int G_Y() { return G_y; }
	inline int G_Z() { return G_z; }
	//-! Get ZNCC
	inline float ZNCC() { return m_fZNCC; }
	//-! Get P0, P
	std::vector<float> P0() { return m_vP0; }
	std::vector<float>  P() { return m_vP; }
	std::vector<float> DP() { return m_vDP; }
	//-! Get iIteration number to compute one POI
	inline int Iteration() { return m_iIteration; }
	//-! Get Conditional parameters
	inline int isDarkSubset() { return m_iDarkSubset; }
	inline int isOutofROI() { return m_iOutofROI; }
	inline int isInvertibleHessian() { return m_iInvertibleHessian; }
	inline int GetProcessed() { return m_iProcessed; }
	inline double GetFFTCCTime() { return m_dFFTCCTime; }
	inline double GetICGNTime() { return m_dICGNTime; }
	inline double GetRangeGood() { return m_iRangeGood; }
	inline int GetEdge() { return m_iEdge; }
	inline int GetEmpty() { return m_iEmpty; }
	inline int GetConverge() { return m_iConverge; }
	void GetAffineMatrix(float *affine_) {
		for (int i = 0; i < 12; ++i) 
			affine_[i] = m_fAffine[i]; 
	}

	//-!Setters
	//-! Set x,y,z values of a POI
	void SetXYZ(int x_, int y_, int z_)
	{
		x = x_;
		y = y_;
		z = z_;
	}
	//-! Set Global x,y,z values of a POI
	void SetGlobalXYZ(int x_, int y_, int z_)
	{
		G_x = x_;
		G_y = y_;
		G_z = z_;
	}
	//!- Set values for P0, and P
	void SetP0(const std::vector<float> &vP0_)
	{
		m_vP0 = vP0_;
	}
	void SetP(const std::vector<float> &vP_)
	{
		m_vP = vP_;
	}
	void SetDP(const std::vector<float> &vDP_)
	{
		m_vDP = vDP_;
	}
	//-! Set ZNCC
	void SetZNCC(float fZNCC_)
	{
		m_fZNCC = fZNCC_;
	}
	void SetFFTCCTime(double FFTCCTime_)
	{
		m_dFFTCCTime = FFTCCTime_;
	}
	void SetICGNTime(double ICGNTime_)
	{
		m_dICGNTime = ICGNTime_;
	}

	//-! Set value for iIteration
	void SetIteration(int iIteration_)
	{
		m_iIteration = iIteration_;
	}
	//-! Set Conditional parameters
	void SetDarkSubset(int iDarkSubset_)
	{
		m_iDarkSubset = iDarkSubset_;
	}
	void SetOutofROI(int iOutofROI_)
	{
		m_iOutofROI = iOutofROI_;
	}
	void SetInvertibleHessian(int iInvertibleHessian_)
	{
		m_iInvertibleHessian = iInvertibleHessian_;
	}
	void SetProcessed(int bP_)
	{
		m_iProcessed = bP_;
	}
	void SetRangeGood(int RangeGood_)
	{
		m_iRangeGood = RangeGood_;
	}

	//affine
	void SetAffineMatrix(float * affine_) {
		for (int i = 0; i < 12; ++i)
			m_fAffine[i] = affine_[i];
	}
	void SetEdge(int iEdge_) {
		m_iEdge = iEdge_;
	}
	void SetEmpty(int iEmpty_) {
		m_iEmpty = iEmpty_;
	}
	void SetConverge(int iConverge_) {
		m_iConverge = iConverge_;
	}
	void SetStrategy(InitStrategy stra) {
		strategy = stra;
	}
	void SetSearchRadius(float rdi) {
		search_radius = rdi;
	}

private:
	int x, y, z;				// Local Position of each POI
	int G_x, G_y, G_z;			// Global Position
	int m_iProcessed;			// whether the POI is processed (1) or not (0), fail(-1), processing(2)
	float m_fZNCC;				// ZNCC
	int m_iIteration;			// iteration num to calculate a POI
	int m_iDarkSubset;			// Zero intensity of all points in a subset
	int m_iEdge;			    // whether the POI is close to edge before ICGN, 0 means not , 1 means yes.
	int m_iOutofROI;			// Calculated displacement is out of ROI
	int m_iInvertibleHessian;	// Hessian matrix is inertable or not
	double m_dFFTCCTime;
	double m_dICGNTime;
	std::vector<float> m_vP0;	// P0 = [u,0,0,0,v,0,0,0,w,0,0,0] 
	std::vector<float> m_vP;	// P = [u,ux,uy,uz,v,vx,vy,vz,w,wx,wy,wz]
	std::vector<float> m_vDP;	// DP= [du,dux,duy,duz,dv,dvx,dvy,dvz,dw,dwx,dwy,dwz]
	int m_iRangeGood;  //Enough keypoint pair inside the range or not
	int m_iEmpty;
	int m_iConverge;

public:
	float m_fAffine[12];
	float GrayValue; //debug
	float Dist = -1.0f;

	std::vector<int> s_idx;
	//candidate
	std::vector<CPUSIFT::Cvec> c_ref;
	std::vector<CPUSIFT::Cvec> c_tar;
	//final
	std::vector<CPUSIFT::Cvec> f_ref;
	std::vector<CPUSIFT::Cvec> f_tar;
	//num of Neighbour kp
	int num_candidate = -1;
	int num_final = -1;
	InitStrategy strategy;
	float search_radius = 0.0;
	int neighbor[6];
};

struct NPOI {
	int POI_id;
	float NeighbourZNCC;
};

/*
	for generating POIs
*/
class POIManager {
public:
	enum generateType {
		uniform = 1, //uniform POI matrix
		ioPreset = 2 //read from file
	};

	//uniform grid generated by parameters, XYZ
	static std::vector<CPOI> uniformGenerator(const int_3d startLoc, const int_3d gradientMargin, const int_3d gridSpace, const int_3d num );
	
	static void uniformNeighborAssign(std::vector<CPOI> &vpoi, const int_3d num, const int num_threads_ = 1);

	//POI locations read from files
	static std::vector<CPOI> ioGenerator(const std::string filename, const int_3d gradientMargin, const bool initialFlag = false);
};


struct NPOI_p_Compare_Neighbour_ZNCC
{
	bool operator()(NPOI*&p1, NPOI*&p2) {
		return p1->NeighbourZNCC < p2->NeighbourZNCC;
	}
};
#endif // !_CPOI_H_