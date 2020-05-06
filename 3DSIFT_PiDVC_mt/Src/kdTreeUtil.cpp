#include "../Include/kdTreeUtil.h"

#include <iostream>
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

using CPUSIFT::Cvec;

//-! 3-1.Computation Tools funs
void KD_Build(kdtree *&_kd, int *&_idx, const vector<Cvec> &vc) {

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

void KD_Destroy(kdtree *&_kd, int *&_idx) {
	if (_idx)
		delete[]_idx;
	if (_kd)
		kd_free(_kd);;
	_kd = nullptr;
	_idx = nullptr;
}

void KD_RangeSerach(kdtree* _kd, vector<Cvec> &neighbor, vector<int> &idx, const Cvec query, const double range) {
	if (range < 0) {
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

void KD_KNNSerach(kdtree* _kd, vector<Cvec> &neighbor, vector<int> &idx, const Cvec query, const int K) {
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
