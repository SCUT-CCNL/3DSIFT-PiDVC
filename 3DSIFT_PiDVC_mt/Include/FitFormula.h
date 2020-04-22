#ifndef _CPU_FIT_FORMULA_H_
#define _CPU_FIT_FORMULA_H_

#include<vector>
#include<array>
#include<random>

//#include"SIFT3D.h"
#include "3DSIFT\Inculde\CSIFT\cSIFT3D.h"

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

#endif // !_REG_H_