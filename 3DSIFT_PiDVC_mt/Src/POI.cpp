#include "POI.h"
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#include "util.h"

using namespace std;

vector<CPOI> POIManager::uniformGenerator(const int_3d startLoc, const int_3d gradientMargin, const int_3d gridSpace, const int_3d num)
{

	int nx = num.x, ny = num.y, nz = num.z;
	int sx = startLoc.x, sy = startLoc.y, sz = startLoc.z;
	int gridX = gridSpace.x, gridY = gridSpace.y, gridZ = gridSpace.z;

	vector<CPOI> vPOI(nx*ny*nz);
	int i = 0;
	for (int curGZ = sz; curGZ < sz + nz*gridZ; curGZ += gridZ) {
		for (int curGY = sy; curGY < sy + ny*gridY; curGY += gridY) {
			for (int curGX = sx; curGX < sx + nx*gridX; curGX += gridX) {
				vPOI[i].SetGlobalXYZ(curGX, curGY, curGZ);
				vPOI[i].SetXYZ(curGX - gradientMargin.x, curGY - gradientMargin.y, curGZ - gradientMargin.z);
				++i;
			}
		}
	}
	return vPOI;
}

void POIManager::uniformNeighborAssign(vector<CPOI> &vpoi, const int_3d num, const int num_threads_) {
	
#pragma omp parallel for num_threads(num_threads_)
	for (int i = 0; i < num.z; ++i) {
		for (int j = 0; j < num.y; ++j) {
			for (int k = 0; k < num.x; ++k) {
				
				int id = ELT(num.y, num.x, k, j, i);
				CPOI &_POI = vpoi[id];

				_POI.neighbor[0] = (k - 1 >= 0) ? (ELT(num.y, num.x, (k - 1), j, i)) : (-1);
				_POI.neighbor[1] = (k + 1<num.x) ? (ELT(num.y, num.x, (k + 1), j, i)) : (-1);

				_POI.neighbor[2] = (j - 1 >= 0) ? (ELT(num.y, num.x, k, (j - 1), i)) : (-1);
				_POI.neighbor[3] = (j + 1<num.y) ? (ELT(num.y, num.x, k, (j + 1), i)) : (-1);

				_POI.neighbor[4] = (i - 1 >= 0) ? (ELT(num.y, num.x, k, j, (i - 1))) : (-1);
				_POI.neighbor[5] = (i + 1<num.z) ? (ELT(num.y, num.x, k, j, (i + 1))) : (-1);
			}
		}
	}
	return;
}

vector<CPOI> POIManager::ioGenerator(const string filename, const int_3d gradientMargin, const bool initialFlag) {

	vector<CPOI> vPOI;
	ifstream read;
	const char deli = ',';
	read.open(filename, ifstream::in);
	if (!read.is_open()) {
		cerr << "Error, Fail to open file:" << filename << endl;
		return vPOI;
	}

	string line;
	int lid = 0;
	while (getline(read, line)) {
		++lid;
		if (line.empty())
			continue;

		int Loc[3];
		float P0[12];

		auto startItera = line.begin();
		auto endItera = line.end();

		for (int i = 0; i < 3; ++i) {
			auto itera = find(startItera, endItera, deli);
			if (itera == line.end() && i!=2) {
				cerr << "Error, NonCorrect format at line" << lid << "'" << line << "'" << " in file:" << filename << endl;
				continue;
			}
			Loc[i] = stoi(string(startItera, itera));
			if(itera != endItera)
				startItera = itera + 1;
		}

		//
		if (initialFlag) {
			for (int i = 0; i < 12; ++i) {
				auto itera = find(startItera, endItera, deli);
				if (itera == line.end() && i != 11) {
					cerr << "Error, NonCorrect format at line" << lid << "'" << line << "'" << " in file:" << filename << endl;
					continue;
				}
				P0[i] = stod(string(startItera, itera));
				if (itera != endItera)
					startItera = itera + 1;
			}
		}

		int GX = Loc[0], GY = Loc[1], GZ = Loc[2];
		CPOI tPOI(GX - gradientMargin.x, GY - gradientMargin.y, GZ- gradientMargin.z);
		tPOI.SetGlobalXYZ(GX, GY, GZ);
		vector<float> vP0 = {
			P0[0] , P0[3], P0[4], P0[5],
			P0[1] , P0[6], P0[7], P0[8],
			P0[2] , P0[9], P0[10], P0[11]
		};
		tPOI.SetP0(vP0);
		tPOI.strategy = Preset;

		vPOI.push_back(tPOI);
	}
	read.close();
	return vPOI;
}
