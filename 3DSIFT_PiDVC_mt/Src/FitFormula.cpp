#include "../Include/FitFormula.h"

#include<iostream>
#include<sstream>
#include<fstream>
#include<array>
#include<random>
#include<ctime>
#include<cfloat>
#include<omp.h>

#include "../Include/kdTreeUtil.h"

#include<Eigen/core>
#include<Eigen/dense>

using namespace CPUSIFT;
using namespace std;

inline void CvecSubtract(Cvec &c1, const Cvec &c2) {
	c1.x -= c2.x;
	c1.y -= c2.y;
	c1.z -= c2.z;
}

void TransferLocalCoor(vector<Cvec> &vc, const Cvec origin) {
	for (auto &ci : vc)
		CvecSubtract(ci, origin);
}

inline float dist_square_Cvec(const Cvec &c1, const Cvec &c2) {
	float dx = c1.x - c2.x;
	float dy = c1.y - c2.y;
	float dz = c1.z - c2.z;
	return dx * dx + dy * dy + dz * dz;
}

inline float dist_Cvec(const Cvec &c1, const Cvec &c2) {
	return sqrtf(dist_square_Cvec(c1, c2));
}


float maxDistanceSquareTo(const vector<Cvec> &vc, const Cvec &target) {
	float maxDistSq = 0.0;
	for (auto ci : vc) {
		float dist_sq = dist_square_Cvec(ci, target);
		if (dist_sq > maxDistSq)
			maxDistSq = dist_sq;
	}
	return maxDistSq;
}

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

//

void siftGuess::init(
	const int threadNum,
	const std::vector<CPUSIFT::Cvec> &vRefPoint,
	const std::vector<CPUSIFT::Cvec> &vTarPoint) {

	thread_num = threadNum;

	_ransacCompute.init_random(thread_num);
	Ref_point = vRefPoint;
	Tar_point = vTarPoint;

}


void siftGuess::setParameters(const int minNeighNum, const float maxExpandRatio, const float errorEpsilon, const int maxIter) {
	m_iMinNeighbor = minNeighNum;
	m_fExpandRatio = maxExpandRatio;
	_ransacCompute.setParam(errorEpsilon, maxIter);
}

void siftGuess::preCompute() {
	KD_Build(kd, kd_idx, Ref_point);
}

void siftGuess::free() {
	KD_Destroy(kd, kd_idx);
}

void siftGuess::compute(CPOI &POI_) {

	//Get global coordinate
	int x = POI_.G_X();
	int y = POI_.G_Y();
	int z = POI_.G_Z();


	//Set range
	double range = sqrt(m_iSubsetX*m_iSubsetX + m_iSubsetY * m_iSubsetY + m_iSubsetZ * m_iSubsetZ) + 0.01;//experiment
	int enough = 1;

	//transform the result into vector of cvec to get the affine transformation matrix and initial guess
	vector<Cvec> tmp_ref, tmp_tar;
	vector<int> tmp_index;

	KD_RangeSerach(kd, tmp_ref, tmp_index, Cvec(x, y, z), range);
	for (auto i : tmp_index)
		tmp_tar.push_back(Tar_point[i]);
	TransferLocalCoor(tmp_ref, Cvec(x, y, z));
	TransferLocalCoor(tmp_tar, Cvec(x, y, z));

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
				fabsf(tmp_ref[i].z) < z_border) {
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
	}
	else {
		// Not Enough KP in subvolume or circumscribed ball
		// Auto extending
		// KNN search is performed
		tmp_ref.clear();
		tmp_tar.clear();
		tmp_index.clear();
		const float up_radius_square = m_fExpandRatio * m_fExpandRatio * range * range;

		KD_KNNSerach(kd, tmp_ref, tmp_index, Cvec(x, y, z), m_iMinNeighbor);
		TransferLocalCoor(tmp_ref, Cvec(x, y, z));
		float maxDist = maxDistanceSquareTo(tmp_ref, Cvec(0, 0, 0));

		if (maxDist > up_radius_square) {
			//searching out of max range
			enough = 0;
			POI_.SetRangeGood(enough);
			POI_.SetEmpty(1);
			return;
		}
		else {
			for (auto i : tmp_index)
				tmp_tar.push_back(Tar_point[i]);
			TransferLocalCoor(tmp_tar, Cvec(x, y, z));
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

	_ransacCompute.Ransac(tmp_ref, tmp_tar, POI_.m_fAffine, POI_.f_ref, POI_.f_tar);

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
	tempP0[9] = POI_.m_fAffine[2];
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