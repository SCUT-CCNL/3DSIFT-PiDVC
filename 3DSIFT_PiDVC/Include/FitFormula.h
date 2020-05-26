#ifndef __PIDVC_FIT_FORMULA_H__
#define __PIDVC_FIT_FORMULA_H__

#include<vector>
#include<array>
#include<random>

#include <kdtree.h>

#include "compute.h"
#include <3DSIFT/Include/cSIFT3D.h>
#include "POI.h"

#define MIN_NEIGHBOUR_KP_NUM 16
#define EPSILON 0.01

//--------------------------------------------------------------------------------

//typedef struct coorNdesc
//{
//	int match_idx;//set -1 at first
//	float rx, ry, rz;
//	float desc[DESC_LENGTH];
//}keyPoint;


//--------------------------------------------------------------------------------

struct mulFitFormula {

	float ransacErrorEpsilon = 3.0f;
	int ransacMaxIterNum = 48;
	float ransacErrorSquare;

	mulFitFormula() { ransacErrorSquare = ransacErrorEpsilon*ransacErrorEpsilon; };
	mulFitFormula(const float error_, const int maxIterNum_):ransacErrorEpsilon(error_), ransacMaxIterNum(maxIterNum_){
		ransacErrorSquare = ransacErrorEpsilon*ransacErrorEpsilon;
	};

	void setParam(const float _errorEpsilon, const int _maxIter) {
		ransacErrorEpsilon = _errorEpsilon;
		ransacMaxIterNum = _maxIter;
		ransacErrorSquare = ransacErrorEpsilon*ransacErrorEpsilon;
	};

	void init_random(int num);

	int Ransac(std::vector<CPUSIFT::Cvec>& points1, std::vector<CPUSIFT::Cvec>& points2,
		float * affine, std::vector<CPUSIFT::Cvec>& src_final, std::vector<CPUSIFT::Cvec>& tar_final);

	int LeastSquares(std::vector<CPUSIFT::Cvec>& points1, std::vector<CPUSIFT::Cvec>& points2, std::vector<int> &consistentIdx,
		float * affine, std::vector<CPUSIFT::Cvec>& src_final, std::vector<CPUSIFT::Cvec>& tar_final);

private:
	std::vector<std::default_random_engine> vRandomEngine;
	std::vector<std::uniform_int_distribution<int>> vIntDistribution;
	std::array<int, 4> get4index(int num);
	float consensus(float * affine, std::vector<CPUSIFT::Cvec>& points1, std::vector<CPUSIFT::Cvec>& points2, std::vector<int>& indices);
};

class siftGuess : public computePOI{
	//ransac compute class
	mulFitFormula _ransacCompute;

	//
	std::vector<CPUSIFT::Cvec> Ref_point;
	std::vector<CPUSIFT::Cvec> Tar_point;
	
	//data structure using kd_tree for fitting Affine
	kdtree *kd = nullptr;
	int *kd_idx = nullptr;

	//parameters
	int m_iMinNeighbor = 16;
	float m_fExpandRatio = 1.2f;

public:
	siftGuess() {};
	~siftGuess() {};
	void init(
		const int threadNum, 
		const std::vector<CPUSIFT::Cvec> &vRefPoint, 
		const std::vector<CPUSIFT::Cvec> &vTarPoint);
	void setParameters(const int minNeighNum, const float maxExpandRatio, const float errorEpsilon, const int maxIter);
	
	void preCompute();
	void free();
	void compute(CPOI &POI_);

	int getMinNeighborNum() { return m_iMinNeighbor; };
	float getExpandRatio() { return m_fExpandRatio; };
	int getRansacIter() { return _ransacCompute.ransacMaxIterNum; };
	int getRansacEpsilon() { return _ransacCompute.ransacErrorEpsilon; };

};

#endif // !_REG_H_
