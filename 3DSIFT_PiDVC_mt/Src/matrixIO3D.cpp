#include <cstring>
#include <cctype>
#include <cmath>
#include "matrixIO3D.h"


//-! writes out the m x n x p matrix onto the stream, in binary format
//-! Row-major
template <class real>
int WriteMatrixToStream(FILE * file, int m, int n, int p, real * matrix)
{
	if ((int)(fwrite(matrix, sizeof(real), m*n*p, file)) < m*n*p)
		return 1;
	return 0;
}
//-! writes out the m x n matrix header onto the stream
int WriteMatrixHeaderToStream(FILE * file, int m, int n, int p)
{
	if (fwrite(&m, sizeof(int), 1, file) < 1)
		return 1;
	if (fwrite(&n, sizeof(int), 1, file) < 1)
		return 1;
	if (fwrite(&p, sizeof(int), 1, file) < 1)
		return 1;

	return 0;
}
//-! Writes matrix to disk binary files
template <class real>
int WriteMatrixToDisk(const char * filename, int m, int n, int p, real * matrix)
{
	FILE * file;
	file = fopen(filename, "wb");
	if (!file)
	{
		printf("Can't open output file: %s.\n", filename);
		return 1;
	}

	if (WriteMatrixHeaderToStream(file, m, n, p) != 0)
	{
		printf("Error writing the matrix header to disk file: %s.\n", filename);
		return 1;
	}

	if (WriteMatrixToStream(file, m, n, p, matrix) != 0)
	{
		printf("Error writing the matrix to disk file: %s.\n", filename);
		return 1;
	}

	fclose(file);

	return 0;
}



//-! Read in the m x n x p matrix into the stream
//-! Row-major
template <class real>
int ReadMatrixFromDisk(const char * filename, int * m, int * n, int *p, real ** matrix)
{
	FILE * file;
	file = fopen(filename, "rb");
	if (!file)
	{
		printf("Can't open input matrix file: %s.\n", filename);
		return 1;
	}

	if (ReadMatrixSizeFromStream(file, m, n, p) != 0)
	{
		printf("Error reading matrix header from disk file: %s.\n", filename);
		return 1;
	}

	//int size = (*m) * (*n) * sizeof(real) + 2 * sizeof(int);
	*matrix = (real *)malloc(sizeof(real)*(*m)*(*n)*(*p));

	if (ReadMatrixFromStream(file, *m, *n, *p, *matrix) != 0)
	{
		printf("Error reading matrix data from disk file: %s.\n", filename);
		return 1;
	}

	fclose(file);

	return 0;
}
int ReadMatrixSizeFromStream(FILE * file, int * m, int * n, int *p)
{
	if (fread(m, sizeof(int), 1, file) < 1)
		return 1;
	if (fread(n, sizeof(int), 1, file) < 1)
		return 1;
	if (fread(p, sizeof(int), 1, file) < 1)
		return 1;

	return 0;
}
//-! read the m x n x p matrix from the stream, in binary format
template <class real>
int ReadMatrixFromStream(FILE * file, int M, int N, int P, real * matrix)
{
	unsigned int readBytes;
	if ((readBytes = fread(matrix, sizeof(real), M*N*P, file)) < (unsigned int)M*N*P)
	{
		printf("Error: I have only read %u bytes. sizeof(real)=%lu\n", readBytes, sizeof(real));
		return 1;
	}

	return 0;
}


//-! Inverse a matrix using Gaussian Jordan Elimination
template<class real>
int InverseMatrix_GaussianJordan(real **&Matrix, real **&invMatrix, int size)
{
	int k, l, m, n;
	int iTemp;
	real dTemp;

	for (l = 0; l < size; l++)
	{
		for (m = 0; m < size; m++)
		{
			if (l == m)
				invMatrix[l][m] = 1;
			else
				invMatrix[l][m] = 0;
		}
	}

	for (l = 0; l < size; l++)
	{
		//-! Find pivot (maximum lth column element) in the rest (6-l) rows
		iTemp = l;
		for (m = l + 1; m < size; m++)
		{
			if (Matrix[m][l] > Matrix[iTemp][l])
			{
				iTemp = m;
			}
		}
		if (fabs(Matrix[iTemp][l]) == 0)
		{
			return 1;
		}
		//-! Swap the row which has maximum lth column element
		if (iTemp != l)
		{
			for (k = 0; k < size; k++)
			{
				dTemp = Matrix[l][k];
				Matrix[l][k] = Matrix[iTemp][k];
				Matrix[iTemp][k] = dTemp;

				dTemp = invMatrix[l][k];
				invMatrix[l][k] = invMatrix[iTemp][k];
				invMatrix[iTemp][k] = dTemp;
			}
		}
		//-! Perform row operation to form required identity matrix out of the Hessian matrix
		for (m = 0; m < size; m++)
		{
			dTemp = Matrix[m][l];
			if (m != l)
			{
				for (n = 0; n < size; n++)
				{
					invMatrix[m][n] -= invMatrix[l][n] * dTemp / Matrix[l][l];
					Matrix[m][n] -= Matrix[l][n] * dTemp / Matrix[l][l];
				}
			}
			else
			{
				for (n = 0; n < size; n++)
				{
					invMatrix[m][n] /= dTemp;
					Matrix[m][n] /= dTemp;
				}
			}
		}
	}
	return 0;
}


//-! Print Matrix to command line
//template <class Real>
//void PrintMatrixInMatlabFormat(int m, int n, int p, Real * U)
//{
//	for (int k = 0; k < p; k++) {
//		for (int i = 0; i<n; i++)
//		{
//			printf("{");
//			for (int j = 0; j < m; j++)
//			{
//				printf("%f", U[ELT(n, p, j, i, k)]);
//				if (j != m - 1)
//					printf(", ");
//			}
//			printf("}");
//
//			if (i != n - 1)
//				printf(",\n");
//		}
//
//		printf("\n}\n");
//	}
//}



template int WriteMatrixToStream<double>(FILE * file, int m, int n, int p, double * matrix);
template int WriteMatrixToStream<float>(FILE * file, int m, int n, int p, float * matrix);
template int WriteMatrixToDisk<double>(const char * filename, int m, int n, int p, double * matrix);
template int WriteMatrixToDisk<float>(const char * filename, int m, int n, int p, float * matrix);

template int ReadMatrixFromStream<double>(FILE * file, int m, int n, int p, double * matrix);
template int ReadMatrixFromStream<float>(FILE * file, int m, int n, int p, float * matrix);
template int ReadMatrixFromDisk<double>(const char * filename, int * m, int * n, int *p, double ** matrix);
template int ReadMatrixFromDisk<float>(const char * filename, int * m, int * n, int *p, float ** matrix);

template int InverseMatrix_GaussianJordan<float>(float **&Matrix, float **&invMatrix, int size);
template int InverseMatrix_GaussianJordan<double>(double **&Matrix, double **&invMatrix, int size);

//template void PrintMatrixInMatlabFormat<double>(int m, int n, int p, double * U);
//template void PrintMatrixInMatlabFormat<float>(int m, int n, int p, float * U);