#pragma once

#include <math.h>
#include <omp.h>
//#include "util.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef MIN
#define MIN(x,y) 	( ((x)<(y)) ? (x):(y) )
#endif
#ifndef MAX
#define MAX(x,y) 	( ((x)>(y)) ? (x):(y) )
#endif

template <class real>
int WriteMatrixHeaderToStream(FILE * file, const int M, const int N, const int P) {

	if (fwrite(&M, sizeof(int), 1, file) < 1)
		return -1;
	if (fwrite(&N, sizeof(int), 1, file) < 1)
		return -1;
	if (fwrite(&P, sizeof(int), 1, file) < 1)
		return -1;

	return 0;
}

template <class real>
int WriteMatrixToStream(FILE * file, const int M, const int N, const int P, const real * Voldata) {

	unsigned int readBytes;
	if ((readBytes = fwrite(Voldata, sizeof(real), M*N*P, file)) < (unsigned int)M*N*P)
	{
		perror("Writing unmatched size of data.\n");
		return -1;
	}

	return 0;
};

template <class real>
class Volume {
public:
	Volume(const int width = 0 , const int height = 0 , const int depth = 0, real *ptr = nullptr) {
		VolWidth = width;
		VolHeight = height;
		VolDepth = depth;
		VolData = ptr;
	}

	~Volume() {
	}

	void Set_WDP(const int width, const int height, const int depth) {
		VolWidth = width;
		VolHeight = height;
		VolDepth = depth;
	}

	int VolWidth;
	int VolHeight;
	int VolDepth;

	real* VolData = nullptr;

	//
	int WriteToDisk(const char* filename) {

		FILE *file = fopen(filename, "wb");
		if (!file) {
			perror("File opening failed \n");
			return -1;
		}

		if (WriteMatrixHeaderToStream<real>(file, VolWidth, VolHeight, VolDepth) != 0)
		{
			perror("Error writing matrix header to disk file.\n");
			return -1;
		}

		if (WriteMatrixToStream<real>(file, VolWidth, VolHeight, VolDepth, VolData) != 0) {
			perror("Error writing matrix data to disk file.\n");
			return -1;
		}

		fclose(file);

		return 0;

	};



};



//Definition
template <class real>
int transpose_xy(real *Vdata, real *VTdata, const int Width, const int Height, const int Depth) {

#pragma omp parallel for
	for (int z = 0; z<Depth; ++z) {
		for (int y = 0; y<Height; ++y) {
			int base = z*(Width*Height) + y*Width;
			for (int x = 0; x<Width; ++x) {
				int trans_id = z*(Width*Height) + x*Height + y;
				VTdata[trans_id] = Vdata[base + x];
			}
		}
	}

	return 0;
}

template <class real>
int transpose_xz(real *Vdata, real *VTdata, const int Width, const int Height, const int Depth) {

#pragma omp parallel for
	for (int z = 0; z<Depth; ++z) {
		for (int y = 0; y<Height; ++y) {
			for (int x = 0; x<Width; ++x) {
				int ori_id = z*(Width*Height) + y*Width + x;
				int trans_id = x*(Height*Depth) + y*Depth + z;
				VTdata[trans_id] = Vdata[ori_id];
			}
		}
	}

	return 0;
}

template <class real>
int transpose_vol_xy(Volume<real> *VData, Volume<real> *VTData) {

	if (VData->VolWidth * VData->VolHeight * VData->VolDepth != VTData->VolWidth * VTData->VolHeight * VTData->VolDepth) {
		perror("Failed in transposing volume because unequal size.\n");
		return -1;
	}

	transpose_xy(VData->VolData, VTData->VolData, VData->VolWidth, VData->VolHeight, VData->VolDepth);

	VTData->VolDepth = VData->VolDepth;
	VTData->VolHeight = VData->VolWidth;
	VTData->VolWidth = VData->VolHeight;

	return 0;
}

template <class real>
int transpose_vol_xz(Volume<real> *VData, Volume<real> *VTData) {

	if (VData->VolWidth * VData->VolHeight * VData->VolDepth != VTData->VolWidth * VTData->VolHeight * VTData->VolDepth) {
		perror("Failed in transposing volume because unequal size.\n");
		return -1;
	}

	transpose_xz(VData->VolData, VTData->VolData, VData->VolWidth, VData->VolHeight, VData->VolDepth);

	VTData->VolDepth = VData->VolWidth;
	VTData->VolHeight = VData->VolHeight;
	VTData->VolWidth = VData->VolDepth;

	return 0;

}


static const float BSplinePreFilter[8] = {
	1.732176555412859f,  //b0
	-0.464135309171000f, //b1
	0.124364681271139f,
	-0.033323415913556f,
	0.008928982383084f,
	-0.002392513618779f,
	0.000641072092032f,
	-0.000171774749350f,  //b7
};

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
template <class real>
inline real w0_c(real a)
{
	//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

template <class real>
inline real w1_c(real a)
{
	//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
	return (1.0f / 6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

template <class real>
inline real w2_c(real a)
{
	//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

template <class real>
inline real w3_c(real a)
{
	return (1.0f / 6.0f)*(a*a*a);
}




template <class real>
int FIR_1D(real *Vinput, real *Voutput, const int length, const int num) {

#pragma omp parallel for 
	for (int i = 0; i<num; ++i) {
		real *base = Vinput + i*length;
		for (int j = 7; j<length - 7; ++j) {
			Voutput[i*length + j] =
				BSplinePreFilter[7] * (base[j - 7] + base[j + 7]) +
				BSplinePreFilter[6] * (base[j - 6] + base[j + 6]) +
				BSplinePreFilter[5] * (base[j - 5] + base[j + 5]) +
				BSplinePreFilter[4] * (base[j - 4] + base[j + 4]) +
				BSplinePreFilter[3] * (base[j - 3] + base[j + 3]) +
				BSplinePreFilter[2] * (base[j - 2] + base[j + 2]) +
				BSplinePreFilter[1] * (base[j - 1] + base[j + 1]) +
				BSplinePreFilter[0] * (base[j]);
		}


		for (int j = 0; j<7; ++j) {

			Voutput[i*length + j] =
				BSplinePreFilter[7] * (base[MAX(j - 7, 0)] + base[j + 7]) +
				BSplinePreFilter[6] * (base[MAX(j - 6, 0)] + base[j + 6]) +
				BSplinePreFilter[5] * (base[MAX(j - 5, 0)] + base[j + 5]) +
				BSplinePreFilter[4] * (base[MAX(j - 4, 0)] + base[j + 4]) +
				BSplinePreFilter[3] * (base[MAX(j - 3, 0)] + base[j + 3]) +
				BSplinePreFilter[2] * (base[MAX(j - 2, 0)] + base[j + 2]) +
				BSplinePreFilter[1] * (base[MAX(j - 1, 0)] + base[j + 1]) +
				BSplinePreFilter[0] * (base[j]);

			/*
			//using left mirror boundary
			Voutput[i*length + j] =
			BSplinePreFilter[7] * (base[LEFT_B(j - 7)] + base[j + 7]) +
			BSplinePreFilter[6] * (base[LEFT_B(j - 6)] + base[j + 6]) +
			BSplinePreFilter[5] * (base[LEFT_B(j - 5)] + base[j + 5]) +
			BSplinePreFilter[4] * (base[LEFT_B(j - 4)] + base[j + 4]) +
			BSplinePreFilter[3] * (base[LEFT_B(j - 3)] + base[j + 3]) +
			BSplinePreFilter[2] * (base[LEFT_B(j - 2)] + base[j + 2]) +
			BSplinePreFilter[1] * (base[LEFT_B(j - 1)] + base[j + 1]) +
			BSplinePreFilter[0] * (base[j]);
			*/
		}

		for (int j = length - 7; j<length; ++j) {

			Voutput[i*length + j] =
				BSplinePreFilter[7] * (base[j - 7] + base[MIN(j + 7, length - 1)]) +
				BSplinePreFilter[6] * (base[j - 6] + base[MIN(j + 6, length - 1)]) +
				BSplinePreFilter[5] * (base[j - 5] + base[MIN(j + 5, length - 1)]) +
				BSplinePreFilter[4] * (base[j - 4] + base[MIN(j + 4, length - 1)]) +
				BSplinePreFilter[3] * (base[j - 3] + base[MIN(j + 3, length - 1)]) +
				BSplinePreFilter[2] * (base[j - 2] + base[MIN(j + 2, length - 1)]) +
				BSplinePreFilter[1] * (base[j - 1] + base[MIN(j + 1, length - 1)]) +
				BSplinePreFilter[0] * (base[j]);


			//using right mirror boundary 
			/*
			Voutput[i*length + j] =
			BSplinePreFilter[7] * (base[j - 7] + base[RIGHT_B(j + 7, length - 1)]) +
			BSplinePreFilter[6] * (base[j - 6] + base[RIGHT_B(j + 6, length - 1)]) +
			BSplinePreFilter[5] * (base[j - 5] + base[RIGHT_B(j + 5, length - 1)]) +
			BSplinePreFilter[4] * (base[j - 4] + base[RIGHT_B(j + 4, length - 1)]) +
			BSplinePreFilter[3] * (base[j - 3] + base[RIGHT_B(j + 3, length - 1)]) +
			BSplinePreFilter[2] * (base[j - 2] + base[RIGHT_B(j + 2, length - 1)]) +
			BSplinePreFilter[1] * (base[j - 1] + base[RIGHT_B(j + 1, length - 1)]) +
			BSplinePreFilter[0] * (base[j]);
			*/
		}
	}
	return 0;
}


template <class real>
int Prefilter(real *Data, real *Coeffi, const int width , const int height, const int depth) {

	Volume<real> *VData = new Volume<real>(width, height, depth, Data);
	Volume<real> *VCoeffi = new Volume<real>(width, height, depth, Coeffi);

	real *p_tmp = (real *)malloc(sizeof(real) * width * height * depth);
	Volume<real> Vtmp(width, height, depth, p_tmp);
	
	//prefilter along x-axis direction
	FIR_1D(VData->VolData, VCoeffi->VolData, VData->VolWidth, VData->VolHeight*VData->VolDepth);

	//transpose xy
	transpose_vol_xy(VCoeffi, &Vtmp);
	//prefilter along y-axis direction(transposed)
	FIR_1D(Vtmp.VolData, VCoeffi->VolData, Vtmp.VolWidth, Vtmp.VolHeight*Vtmp.VolDepth);
	//essential due to Width/Height/Depth needed at transpose
	VCoeffi->Set_WDP(Vtmp.VolWidth, Vtmp.VolHeight, Vtmp.VolDepth);
	//transpose yx
	transpose_vol_xy(VCoeffi, &Vtmp);

	//transpose xz
	transpose_vol_xz(&Vtmp, VCoeffi);
	//prefilter along z-axis direction(transposed)
	FIR_1D(VCoeffi->VolData, Vtmp.VolData, VCoeffi->VolWidth, VCoeffi->VolHeight*VCoeffi->VolDepth);
	Vtmp.Set_WDP(VCoeffi->VolWidth, VCoeffi->VolHeight, VCoeffi->VolDepth);
	//transpose zx
	transpose_vol_xz(&Vtmp, VCoeffi);

	free(p_tmp);

	delete VData;
	delete VCoeffi;

	return 0;
}

