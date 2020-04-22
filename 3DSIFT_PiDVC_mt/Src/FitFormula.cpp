#include "FitFormula.h"

#include<iostream>
#include<sstream>
#include<fstream>
#include<array>
#include<random>
#include<ctime>
#include<cfloat>
#include<omp.h>

#include<Eigen/core>
#include<Eigen/dense>

using namespace CPUSIFT;

float error_square(Cvec& ref, Cvec& tar, const float * const affine)
{
	float result = 0.0;
	Cvec tar_cal;

	//使用仿射矩阵算出的坐标
	tar_cal.x = ref.x*affine[0] + ref.y*affine[3] + ref.z*affine[6] + affine[9];
	tar_cal.y = ref.x*affine[1] + ref.y*affine[4] + ref.z*affine[7] + affine[10];
	tar_cal.z = ref.x*affine[2] + ref.y*affine[5] + ref.z*affine[8] + affine[11];

	//仿射坐标与原对应点的坐标之差的平方
	result += (tar.x - tar_cal.x)*(tar.x - tar_cal.x);
	result += (tar.y - tar_cal.y)*(tar.y - tar_cal.y);
	result += (tar.z - tar_cal.z)*(tar.z - tar_cal.z);

	return result;
}

void mulFitFormula::init_random(int num) {
	for (int i = 0; i < num; ++i) {
		vRandomEngine.push_back(std::default_random_engine(time(NULL) + i));
		vIntDistribution.push_back(std::uniform_int_distribution<int>());
	}
}

std::array<int, 4> mulFitFormula::get4index(int num)
{
	int tid = omp_get_thread_num();
	auto &engine = vRandomEngine[tid];
	auto &distri = vIntDistribution[tid];
	std::array<int, 4> result = { 0,1,2,3 };
	for (int i = 0; i < 4; i++)
	{
		result[i] = (distri(engine)) % num;
		for (int k = 0; k < i; k++)
		{
			if (result[i] == result[k])
			{
				i--;
			}
		}
	}
	return result;
}

float mulFitFormula::consensus(float * affine, std::vector<Cvec>& points1, std::vector<Cvec>& points2, std::vector<int>& indices)
{
	float sum_error = 0;
	for (int i = 0; i < points1.size(); i++)
	{
		float error = error_square(points1.at(i), points2.at(i), affine);
		if (error < ransacErrorSquare )
		{
			indices.push_back(i);
			sum_error += error;
		}
	}
	return sum_error;
}

int mulFitFormula::Ransac(std::vector<Cvec>& points1, std::vector<Cvec>& points2, float * affine,
	std::vector<Cvec>& src_final, std::vector<Cvec>& tar_final)
{
	int numOfPoint = points1.size();

	std::vector<int> agreeIndices;
	float best_error = FLT_MAX;

	std::array<int, 4> random_indice; //store random 4 index 
	for (int i = 1; i <= ransacMaxIterNum; i++)
	{
		random_indice = get4index(numOfPoint);

		Eigen::Matrix4d cur_A;
		Eigen::MatrixXd cur_B(4, 3);
		Eigen::MatrixXd cur_affine(4, 3);

		for (int j = 0; j < 4; j++)
		{
			cur_A(j, 0) = points1.at(random_indice[j]).x;
			cur_A(j, 1) = points1.at(random_indice[j]).y;
			cur_A(j, 2) = points1.at(random_indice[j]).z;
			cur_A(j, 3) = 1;
			cur_B(j, 0) = points2.at(random_indice[j]).x;
			cur_B(j, 1) = points2.at(random_indice[j]).y;
			cur_B(j, 2) = points2.at(random_indice[j]).z;
		}

		//Model M1
		cur_affine = cur_A.colPivHouseholderQr().solve(cur_B);
		//cur_affine = cur_A.inverse() * cur_B;


		//double* affine to store the Model for test
		for (int k = 0; k < 4; k++) {
			for (int kk = 0; kk < 3; kk++)
			{
				affine[k * 3 + kk] = cur_affine(k, kk);
			}
		}

		std::vector<int> tempIndices;
		float tempError = consensus(affine, points1, points2, tempIndices);

		if (tempIndices.size()> agreeIndices.size())
		{
			agreeIndices.assign(tempIndices.begin(), tempIndices.end());
			//best_error = tempError;
		}
		//else if (tempIndices.size() == agreeIndices.size() && tempError < best_error) {
		//	agreeIndices.assign(tempIndices.begin(), tempIndices.end());
		//	best_error = tempError;
		//}

	}

	LeastSquares(points1, points2, agreeIndices, affine, src_final, tar_final);

	return 0;
}

int mulFitFormula::LeastSquares(std::vector<Cvec>& points1, std::vector<Cvec>& points2, std::vector<int> &consistentIdx, float * affine, std::vector<Cvec>& src_final, std::vector<Cvec>& tar_final) {

	int consensus_size = consistentIdx.size();

	Eigen::MatrixXd A(consensus_size, 4);
	Eigen::MatrixXd B(consensus_size, 3);

	for (int i = 0; i < consensus_size; i++)
	{
		int id = consistentIdx[i];
		A(i, 0) = points1[id].x;
		A(i, 1) = points1[id].y;
		A(i, 2) = points1[id].z;
		A(i, 3) = 1;

		B(i, 0) = points2[id].x;
		B(i, 1) = points2[id].y;
		B(i, 2) = points2[id].z;

		src_final.push_back(points1[id]);
		tar_final.push_back(points2[id]);
	}

	//X is  Model M1*,using least squares
	/*Eigen::MatrixXd X = (A.transpose() * A).ldlt().solve(A.transpose() * B);*/
	Eigen::MatrixXd X = A.colPivHouseholderQr().solve(B);

	for (int k = 0; k<4; k++)
		for (int kk = 0; kk < 3; kk++)
		{
			affine[k * 3 + kk] = X(k, kk);
		}

	//std::cout << X.transpose() << std::endl << std::endl;

	return 0;
};