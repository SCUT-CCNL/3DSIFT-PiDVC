#ifndef __PIDVC_MATRIXIO3D_H__
#define __PIDVC_MATRIXIO3D_H__

#include <cstdio>
#include <cstdlib>
//#include "matrixMacro3D.h"

// =============== 3D Matrix Operation Rountines =================//
/*
* Read and write volume image from and to disk binary files.       *
* File Format: m n p, data...								        *
* where m, n and p refer to depth, height and width, respectively, *
* data are the intensity values of every voxel.                    *
* NOTE: Matrix are saved as row-major                              *
*/

//!- Read matrix into memory
template <class real>
int ReadMatrixFromDisk(const char * filename, int * m, int * n, int *p, real ** matrix);
int ReadMatrixSizeFromDisk(const char * filename, int * m, int * n, int *p);
template <class real>
int ReadMatrixFromStream(FILE * file, int m, int n, int p, real * matrix);
int ReadMatrixSizeFromStream(FILE * file, int * m, int * n, int *p);

//!- Writes matrix to disk
template <class real>
int WriteMatrixToDisk(const char * filename, int m, int n, int p, real * matrix);
template <class real>
int WriteMatrixToStream(FILE * file, int m, int n, int p, real * matrix);
int WriteMatrixHeaderToStream(FILE * file, int m, int n, int p);

//!- Compute the inverse of matrix using Gaussian Jordan Elimination
template<class real>
int InverseMatrix_GaussianJordan(real **&Matrix, real **&invMatrix, int size);

//!- prints the matrix to standard output in Matlab format
template <class real>
void PrintMatrixInMatlabFormat(int m, int n, int p, real * U);


#endif // !_MATRIXIO3D_H_
