#include "../Include/ICGN.h"

#include <iostream>
#include <omp.h>

#include "../Include/MemManager.h"
#include "../Include/Spline.h"

using namespace std;

void icgnRegistration::init(const int threadNum,
	float *fVolR, float *fVolT,
	int iOriginVolWidth, int iOriginVolHeight, int iOriginVolDepth,
	float ***fRx, float ***fRy, float ***fRz) {

	thread_num = threadNum;
	m_fVolR = fVolR;
	m_fVolT = fVolT;
	m_iOriginVolWidth = iOriginVolWidth;
	m_iOriginVolHeight = iOriginVolHeight;
	m_iOriginVolDepth = iOriginVolDepth;
	m_fRx = fRx;
	m_fRy = fRy;
	m_fRz = fRz;
	
	if (m_iSubsetX <= 0 || m_iSubsetY <= 0 || m_iSubsetZ <= 0) {
		cerr << "Invalid subset radii in icgnRegistration::init()" << endl;
		throw "Invalid Subset Radii";
	}

	const int iSubsetW = 2 * m_iSubsetX + 1;
	const int iSubsetH = 2 * m_iSubsetY + 1;
	const int iSubsetD = 2 * m_iSubsetZ + 1;
	CMemManager<float>::hCreatePtr(m_fRDescent, thread_num, iSubsetD, iSubsetH, iSubsetW, 12);
	CMemManager<float>::hCreatePtr(m_fSubsetR, thread_num, iSubsetD, iSubsetH, iSubsetW);
	CMemManager<float>::hCreatePtr(m_fSubsetT, thread_num, iSubsetD, iSubsetH, iSubsetW);
}

void icgnRegistration::free() {
	CMemManager<float>::hDestroyPtr(m_fSubsetR);
	CMemManager<float>::hDestroyPtr(m_fSubsetT);
	CMemManager<float>::hDestroyPtr(m_fRDescent);
}

void icgnRegistration::compute(CPOI &POI_) {

	const int iMaxIteration = m_iMaxIterationNum;
	const float fDeltaP = m_dDeltaP;
	const int iID_ = omp_get_thread_num();

	POI_.SetProcessed(2);

	//-! Define the size of subvolume window for IC-GN algorithm
	int iSubsetX = m_iSubsetX;
	int iSubsetY = m_iSubsetY;
	int iSubsetZ = m_iSubsetZ;

	int iOriginVolWidth = m_iOriginVolWidth;
	int iOriginVolHeight = m_iOriginVolHeight;
	int iOriginVolDepth = m_iOriginVolDepth;
	int iSubsetW = iSubsetX * 2 + 1;
	int iSubsetH = iSubsetY * 2 + 1;
	int iSubsetD = iSubsetZ * 2 + 1;

	float fSubsetSize = float(iSubsetD*iSubsetH*iSubsetW);
	float ***m_fSubsetR_id = m_fSubsetR[iID_];
	float ***m_fSubsetT_id = m_fSubsetT[iID_];
	float ****m_fRDescent_id = m_fRDescent[iID_];

	int iTempX, iTempY, iTempZ;
	float fSubAveR, fSubAveT, fSubNorR, fSubNorT;
	float fU, fUx, fUy, fUz, fV, fVx, fVy, fVz, fW, fWx, fWy, fWz;
	float fDU, fDUx, fDUy, fDUz, fDV, fDVx, fDVy, fDVz, fDW, fDWx, fDWy, fDWz;
	float fWarpX, fWarpY, fWarpZ, m_fTemp, m_fTempX, m_fTempY, m_fTempZ;
	float fErrors;
	float fZNCC;
	float f00, f10, f20;
	float f01, f11, f21;
	float f02, f12, f22;
	float f03, f13, f23;
	vector<float>P(12, 0);
	vector<float>DP(12, 0);

	vector<float> m_fNumerator(12, 0);
	// 12-vector
	vector<vector<float>> m_fInvHessian(12, vector<float>(12, 0));
	// Inverse Hessian matrix 12x12
	vector<vector<float>> m_fHessian(12, vector<float>(12, 0));
	// Hessian of a POI 12x12
	vector<vector<float>> m_fHessianXYZ(12, vector<float>(12, 0));
	// Hessian of each point around a POI 12x12
	vector<vector<float>> m_fJacobian(3, vector<float>(12, 0));
	// Jacobian matrix deltaR*(dW/dP) 3x12
	vector<vector<float>> m_fWarp(4, vector<float>(4, 0));
	// Warp Function 4x4

	float w_x[4];
	float w_y[4];
	float w_z[4];
	float sum_x[4];
	float sum_y[4];
	//BSPLINE

	//-! Start the timer for IC-GN algorithm

	//w.start();


	//-! Initialize parameters of Subvolume R
	fSubAveR = 0; // R_m
	fSubNorR = 0; // sqrt (Sigma(R_i - R_m)^2)
					//-! Initialize the Hessian matrix for each subvolume
	for (int k = 0; k < 12; k++)
	{
		for (int n = 0; n < 12; n++)
		{
			m_fHessian[k][n] = 0;
		}
	}


	//-! Feed the gray intensity to subvolume R
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			for (int n = 0; n < iSubsetW; n++)
			{
				m_fSubsetR_id[l][m][n] =
					m_fVolR[ELT(m_iOriginVolHeight, m_iOriginVolWidth,
						POI_.G_X() - iSubsetX + n,
						POI_.G_Y() - iSubsetY + m,
						POI_.G_Z() - iSubsetZ + l)];

				fSubAveR += m_fSubsetR_id[l][m][n];

				//-! Evaluate the Jacbian dW/dp at (x, 0), etc. POI
				m_fJacobian[0][0] = 1;
				m_fJacobian[0][1] = float(n - iSubsetX);
				m_fJacobian[0][2] = float(m - iSubsetY);
				m_fJacobian[0][3] = float(l - iSubsetZ);
				m_fJacobian[0][4] = 0;
				m_fJacobian[0][5] = 0;
				m_fJacobian[0][6] = 0;
				m_fJacobian[0][7] = 0;
				m_fJacobian[0][8] = 0;
				m_fJacobian[0][9] = 0;
				m_fJacobian[0][10] = 0;
				m_fJacobian[0][11] = 0;

				m_fJacobian[1][0] = 0;
				m_fJacobian[1][1] = 0;
				m_fJacobian[1][2] = 0;
				m_fJacobian[1][3] = 0;
				m_fJacobian[1][4] = 1;
				m_fJacobian[1][5] = float(n - iSubsetX);
				m_fJacobian[1][6] = float(m - iSubsetY);
				m_fJacobian[1][7] = float(l - iSubsetZ);
				m_fJacobian[1][8] = 0;
				m_fJacobian[1][9] = 0;
				m_fJacobian[1][10] = 0;
				m_fJacobian[1][11] = 0;

				m_fJacobian[2][0] = 0;
				m_fJacobian[2][1] = 0;
				m_fJacobian[2][2] = 0;
				m_fJacobian[2][3] = 0;
				m_fJacobian[2][4] = 0;
				m_fJacobian[2][5] = 0;
				m_fJacobian[2][6] = 0;
				m_fJacobian[2][7] = 0;
				m_fJacobian[2][8] = 1;
				m_fJacobian[2][9] = float(n - iSubsetX);
				m_fJacobian[2][10] = float(m - iSubsetY);
				m_fJacobian[2][11] = float(l - iSubsetZ);

				float *m_fRDescent_id_ilmn = m_fRDescent_id[l][m][n];
				//-! Compute the steepest descent image DealtR*dW/dp
				for (int k = 0; k < 12; k++)
				{
					//m_fRDescent_id[l][m][n][k] =
					m_fRDescent_id_ilmn[k] =
						m_fRx[POI_.Z() - iSubsetZ + l][POI_.Y() - iSubsetY + m][POI_.X() - iSubsetX + n] * m_fJacobian[0][k] +
						m_fRy[POI_.Z() - iSubsetZ + l][POI_.Y() - iSubsetY + m][POI_.X() - iSubsetX + n] * m_fJacobian[1][k] +
						m_fRz[POI_.Z() - iSubsetZ + l][POI_.Y() - iSubsetY + m][POI_.X() - iSubsetX + n] * m_fJacobian[2][k];
				}

				//-! Compute the Hessian matrix
				for (int j = 0; j < 12; j++)
				{
					//float m_fRDescent_lmnj_tmp = m_fRDescent_id[l][m][n][j];
					for (int k = 0; k < 12; k++)
					{
						//-! Hessian matrix at each point
						//Origin Accessing
						//m_fHessianXYZ[j][k] = m_fRDescent_id[l][m][n][j] * m_fRDescent_id[l][m][n][k];
						m_fHessianXYZ[j][k] = m_fRDescent_id_ilmn[j] * m_fRDescent_id_ilmn[k];

						//-! sum of Hessian matrix at all the points in subvolume R
						m_fHessian[j][k] += m_fHessianXYZ[j][k];
					}
				}
			}
		}
	}

	//DEBUG
	POI_.GrayValue = m_fSubsetR_id[iSubsetZ][iSubsetY][iSubsetX];
	//cout << POI_.G_X() << ","<< POI_.G_Y() << "," << POI_.G_Z() << "," << POI_.GrayValue << endl;

	//-! Check if Subset R is a all dark subvolume
	if (fSubAveR == 0)
	{
		POI_.SetDarkSubset(2);
		POI_.SetProcessed(-1);
		return;
	}
	fSubAveR /= fSubsetSize;
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			for (int n = 0; n < iSubsetW; n++)
			{
				m_fSubsetR_id[l][m][n] = m_fSubsetR_id[l][m][n] - fSubAveR;	// R_i - R_m
				fSubNorR += pow(m_fSubsetR_id[l][m][n], 2);				// Sigma(R_i - R_m)^2
			}
		}
	}
	fSubNorR = sqrt(fSubNorR);	 // sqrt (Sigma(R_i - R_m)^2)
	if (fSubNorR == 0)
	{
		POI_.SetDarkSubset(3);
		POI_.SetProcessed(-1);
		return;
	}

	//-! Invert the Hessian matrix (Gauss-Jordan algorithm)
	if (1 == InverseHessian_GaussianJordan(m_fInvHessian, m_fHessian))
	{
		POI_.SetInvertibleHessian(1);
		POI_.SetProcessed(-1);
		return;
	}

	//-! Initialize matrix P and DP
	POI_.SetP(POI_.P0());
	fU = POI_.P()[0];
	fUx = POI_.P()[1];
	fUy = POI_.P()[2];
	fUz = POI_.P()[3];
	fV = POI_.P()[4];
	fVx = POI_.P()[5];
	fVy = POI_.P()[6];
	fVz = POI_.P()[7];
	fW = POI_.P()[8];
	fWx = POI_.P()[9];
	fWy = POI_.P()[10];
	fWz = POI_.P()[11];

	//-! Initialize the warp matrix: W
	m_fWarp[0][0] = 1 + fUx;		m_fWarp[0][1] = fUy;		m_fWarp[0][2] = fUz;		m_fWarp[0][3] = fU;
	m_fWarp[1][0] = fVx;			m_fWarp[1][1] = 1 + fVy;	m_fWarp[1][2] = fVz;		m_fWarp[1][3] = fV;
	m_fWarp[2][0] = fWx;			m_fWarp[2][1] = fWy;		m_fWarp[2][2] = 1 + fWz;	m_fWarp[2][3] = fW;
	m_fWarp[3][0] = 0;				m_fWarp[3][1] = 0;			m_fWarp[3][2] = 0;			m_fWarp[3][3] = 1;

	//-! Initialize DeltaP
	fDU = fDUx = fDUy = fDUz = fDV = fDVx = fDVy = fDVz = fDW = fDWx = fDWy = fDWz = 0;

	//-! Filled warped voxel into Subset T
	fSubAveT = 0;
	fSubNorT = 0;
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
			for (int n = 0; n < iSubsetW; n++)
			{
				//-! Calculate the location of warped subvolume T
				fWarpX =
					POI_.X() + 1 + m_fWarp[0][0] * (n - iSubsetX) + m_fWarp[0][1] * (m - iSubsetY) + m_fWarp[0][2] * (l - iSubsetZ) + m_fWarp[0][3];
				fWarpY =
					POI_.Y() + 1 + m_fWarp[1][0] * (n - iSubsetX) + m_fWarp[1][1] * (m - iSubsetY) + m_fWarp[1][2] * (l - iSubsetZ) + m_fWarp[1][3];
				fWarpZ =
					POI_.Z() + 1 + m_fWarp[2][0] * (n - iSubsetX) + m_fWarp[2][1] * (m - iSubsetY) + m_fWarp[2][2] * (l - iSubsetZ) + m_fWarp[2][3];
				iTempX = int(fWarpX);
				iTempY = int(fWarpY);
				iTempZ = int(fWarpZ);

				m_fTempX = fWarpX - iTempX;
				m_fTempY = fWarpY - iTempY;
				m_fTempZ = fWarpZ - iTempZ;
				// To make sure BSPline coefficient access is valid and avoid too many judges at interpolation
				if ((iTempX >= 1) && (iTempY >= 1) && (iTempZ >= 1) &&
					(iTempX <= (iOriginVolWidth - 3)) && (iTempY <= (iOriginVolHeight - 3)) && (iTempZ <= (iOriginVolDepth - 3)))
				{

					//-! If it is integer-pixel location, feed the gray intensity of T into the subvolume T
					if (m_fTempX == 0 && m_fTempY == 0 && m_fTempZ == 0)
						m_fSubsetT_id_lm[n] = m_fVolT[ELT(m_iOriginVolHeight, m_iOriginVolWidth, iTempX, iTempY, iTempZ)];

					//-! In most case, it is sub-pixel location, estimate the gary intensity using interpolation
					else {

						w_x[0] = w0_c(m_fTempX);
						w_x[1] = w1_c(m_fTempX);
						w_x[2] = w2_c(m_fTempX);
						w_x[3] = w3_c(m_fTempX);

						w_y[0] = w0_c(m_fTempY);
						w_y[1] = w1_c(m_fTempY);
						w_y[2] = w2_c(m_fTempY);
						w_y[3] = w3_c(m_fTempY);

						w_z[0] = w0_c(m_fTempZ);
						w_z[1] = w1_c(m_fTempZ);
						w_z[2] = w2_c(m_fTempZ);
						w_z[3] = w3_c(m_fTempZ);

						m_fSubsetT_id_lm[n] = 0;


						//z
						for (int j = 0; j < 4; ++j) {
							//y
							//due to 2 margin data point are prepared in the array, thus +2 is necessary
							float **m_fTGBspline_j = m_fTGBspline[iTempZ + 2 - 1 + j];
							for (int i = 0; i < 4; ++i) {
								float * m_fTGBspline_ji = m_fTGBspline_j[iTempY + 2 - 1 + i];
								/*sum_x[i] =
									w_x[0] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 - 1]
									+ w_x[1] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1]
									+ w_x[2] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 1]
									+ w_x[3] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 2];*/
								sum_x[i] =
									w_x[0] * m_fTGBspline_ji[iTempX + 2 - 1]
									+ w_x[1] * m_fTGBspline_ji[iTempX + 2]
									+ w_x[2] * m_fTGBspline_ji[iTempX + 2 + 1]
									+ w_x[3] * m_fTGBspline_ji[iTempX + 2 + 2];
							}
							sum_y[j] =
								w_y[0] * sum_x[0]
								+ w_y[1] * sum_x[1]
								+ w_y[2] * sum_x[2]
								+ w_y[3] * sum_x[3];
						}

						m_fSubsetT_id_lm[n] =
							w_z[0] * sum_y[0]
							+ w_z[1] * sum_y[1]
							+ w_z[2] * sum_y[2]
							+ w_z[3] * sum_y[3];

					}
					fSubAveT += m_fSubsetT_id_lm[n];

					//debug
#ifdef _JR_DEBUG
					if (((l*iSubsetH + m)*iSubsetW + n) == 0) {
						printf("id:%d; PRef:(%d,%d,%d); Ptar:(%f,%f,%f); val:%f; iTempX:(%d,%d,%d); inerp:(%f)\n", ((l*iSubsetH + m)*iSubsetW + n), n, m, l, fWarpX, fWarpY, fWarpZ, m_fSubsetT_id_lm[n], iTempX, iTempY, iTempZ, m_fTGBspline[iTempZ + 2][iTempY + 2][iTempX + 2]);
					}
#endif
				}
				else
				{
					POI_.SetOutofROI(1);	// if the loacation of the warped subvolume T is out of the ROI, stop iteration and set p as the current value
					POI_.SetProcessed(-2);
					return;
				}
			}
		}
	}
	fSubAveT /= fSubsetSize;
	//-! Check if Subset T is a all dark subvolume
	if (fSubAveT == 0)
	{
		POI_.SetDarkSubset(4);
		POI_.SetProcessed(-1);
		return;
	}
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			for (int n = 0; n < iSubsetW; n++)
			{
				m_fSubsetT_id[l][m][n] = m_fSubsetT_id[l][m][n] - fSubAveT;	//T_i - T_m
				fSubNorT += pow(m_fSubsetT_id[l][m][n], 2);				// Sigma(T_i - T_m)^2
			}
		}
	}
	fSubNorT = sqrt(fSubNorT);	// sqrt (Sigma(T_i - T_m)^2)

	if (fSubNorT == 0)
	{
		POI_.SetDarkSubset(5);	// Set flag
		POI_.SetProcessed(-1);
		return;
	}

	//-! Compute the error image
	for (int k = 0; k < 12; k++)
	{
		m_fNumerator[k] = 0;
	}
	fZNCC = 0;
	for (int l = 0; l < iSubsetD; l++)
	{
		for (int m = 0; m < iSubsetH; m++)
		{
			float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
			float *m_fSubsetR_id_lm = m_fSubsetR_id[l][m];
			for (int n = 0; n < iSubsetW; n++)
			{
				fErrors = (fSubNorR / fSubNorT) * m_fSubsetT_id_lm[n] - m_fSubsetR_id_lm[n];
				fZNCC += m_fSubsetT_id_lm[n] * m_fSubsetR_id_lm[n];
				//-! Compute the numerator
				float *m_fRDescent_id_lmn = m_fRDescent_id[l][m][n];
				for (int k = 0; k < 12; k++)
				{
					//m_fNumerator[k] += (m_fRDescent_id[l][m][n][k] * fErrors);
					m_fNumerator[k] += (m_fRDescent_id_lmn[k] * fErrors);
				}
			}
		}
	}
	fZNCC = fZNCC / (fSubNorT*fSubNorR);
	POI_.SetZNCC(fZNCC);

#ifdef _JR_DEBUG
	printf("iteration%d, normT:%f, normR/normT:%f\n", POI_.Iteration(), fSubNorT, fSubNorR / fSubNorT);
#endif

	//-! Compute DeltaP
	for (int k = 0; k < 12; k++)
	{
		DP[k] = 0;
		for (int n = 0; n < 12; n++)
		{
			DP[k] += (m_fInvHessian[k][n] * m_fNumerator[n]);
		}
	}
	POI_.SetDP(DP);
	fDU = POI_.DP()[0];
	fDUx = POI_.DP()[1];
	fDUy = POI_.DP()[2];
	fDUz = POI_.DP()[3];
	fDV = POI_.DP()[4];
	fDVx = POI_.DP()[5];
	fDVy = POI_.DP()[6];
	fDVz = POI_.DP()[7];
	fDW = POI_.DP()[8];
	fDWx = POI_.DP()[9];
	fDWy = POI_.DP()[10];
	fDWz = POI_.DP()[11];

	//-! Update the warp
	//-! m_fTemp: store the denominator of the Inverse W
	m_fTemp =
		((fDWz + 1)*fDVy + (-fDWy * fDVz + (fDWz + 1)))*fDUx +
		(((-fDWz - 1)*fDVx + fDWx * fDVz)*fDUy + ((fDWy*fDVx + (-fDWx * fDVy - fDWx))*fDUz +
		((fDWz + 1)*fDVy + (-fDWy * fDVz + (fDWz + 1)))));
	if (m_fTemp == 0)
	{
		POI_.SetInvertibleHessian(2);
		POI_.SetProcessed(-1);
		return;
	}

	//-! W(P) <- W(P) o W(DP)^-1
	f00 = (fDWz + 1)*fDVy + (-fDWy * fDVz + (fDWz + 1));
	f10 = (-fDWz - 1)*fDVx + fDWx * fDVz;
	f20 = fDWy * fDVx + (-fDWx * fDVy - fDWx);
	f01 = (-fDWz - 1)*fDUy + fDWy * fDUz;
	f11 = (fDWz + 1)*fDUx + (-fDWx * fDUz + (fDWz + 1));
	f21 = -fDWy * fDUx + (fDWx*fDUy - fDWy);
	f02 = fDVz * fDUy + (-fDVy - 1)*fDUz;
	f12 = -fDVz * fDUx + (fDVx*fDUz - fDVz);
	f22 = (fDVy + 1)*fDUx + (-fDVx * fDUy + (fDVy + 1));
	f03 = (-fDW * fDVz + (fDWz + 1)*fDV)*fDUy + ((fDW*fDVy + (-fDWy * fDV + fDW))*fDUz + ((-fDWz - 1)*fDVy + (fDWy*fDVz + (-fDWz - 1)))*fDU);
	f13 = (fDW*fDVz + (-fDWz - 1)*fDV)*fDUx + ((-fDW * fDVx + fDWx * fDV)*fDUz + (((fDWz + 1)*fDVx - fDWx * fDVz)*fDU + (fDW*fDVz + (-fDWz - 1)*fDV)));
	f23 = (-fDW * fDVy + (fDWy*fDV - fDW))*fDUx + ((fDW*fDVx - fDWx * fDV)*fDUy + ((-fDWy * fDVx + (fDWx*fDVy + fDWx))*fDU + (-fDW * fDVy + (fDWy*fDV - fDW))));
	m_fWarp[0][0] = ((1 + fUx)*f00 + fUy * f10 + fUz * f20) / m_fTemp;	m_fWarp[0][1] = ((1 + fUx)*f01 + fUy * f11 + fUz * f21) / m_fTemp; 	m_fWarp[0][2] = ((1 + fUx)*f02 + fUy * f12 + fUz * f22) / m_fTemp; 	m_fWarp[0][3] = ((1 + fUx)*f03 + fUy * f13 + fUz * f23) / m_fTemp + fU;
	m_fWarp[1][0] = (fVx*f00 + (1 + fVy)*f10 + fVz * f20) / m_fTemp;	m_fWarp[1][1] = (fVx*f01 + (1 + fVy)*f11 + fVz * f21) / m_fTemp;	m_fWarp[1][2] = (fVx*f02 + (1 + fVy)*f12 + fVz * f22) / m_fTemp;	m_fWarp[1][3] = (fVx*f03 + (1 + fVy)*f13 + fVz * f23) / m_fTemp + fV;
	m_fWarp[2][0] = (fWx*f00 + fWy * f10 + (1 + fWz)*f20) / m_fTemp;	m_fWarp[2][1] = (fWx*f01 + fWy * f11 + (1 + fWz)*f21) / m_fTemp;	m_fWarp[2][2] = (fWx*f02 + fWy * f12 + (1 + fWz)*f22) / m_fTemp;	m_fWarp[2][3] = (fWx*f03 + fWy * f13 + (1 + fWz)*f23) / m_fTemp + fW;
	m_fWarp[3][0] = 0;	m_fWarp[3][1] = 0;	m_fWarp[3][2] = 0;	m_fWarp[3][3] = 1;

	//-! Update P
	P[0] = m_fWarp[0][3];	P[1] = m_fWarp[0][0] - 1;	P[2] = m_fWarp[0][1];		P[3] = m_fWarp[0][2];
	P[4] = m_fWarp[1][3];	P[5] = m_fWarp[1][0];		P[6] = m_fWarp[1][1] - 1;	P[7] = m_fWarp[1][2];
	P[8] = m_fWarp[2][3];	P[9] = m_fWarp[2][0];		P[10] = m_fWarp[2][1];		P[11] = m_fWarp[2][2] - 1;
	POI_.SetP(P);
	fU = POI_.P()[0];
	fUx = POI_.P()[1];
	fUy = POI_.P()[2];
	fUz = POI_.P()[3];
	fV = POI_.P()[4];
	fVx = POI_.P()[5];
	fVy = POI_.P()[6];
	fVz = POI_.P()[7];
	fW = POI_.P()[8];
	fWx = POI_.P()[9];
	fWy = POI_.P()[10];
	fWz = POI_.P()[11];
	POI_.SetIteration(1);

#ifdef _JR_DEBUG
	printf("iteration: %d, deltaP:%f\n", 0, sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)));
	printf("\t\tmatrix:%f %f %f %f, %f %f %f %f, %f %f %f %f \n",
		P[0], P[1], P[2], P[3],
		P[4], P[5], P[6], P[7],
		P[8], P[9], P[10], P[11]);
#endif

	//-! Perform interative optimization, with pre-set maximum iteration step
	while (POI_.Iteration() < iMaxIteration && sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)) > fDeltaP)
	{
		POI_.SetIteration(POI_.Iteration() + 1);

		//-! Fill warpped image into Subset T
		fSubAveT = 0;
		fSubNorT = 0;
		for (int l = 0; l < iSubsetD; l++)
		{
			for (int m = 0; m < iSubsetH; m++)
			{
				float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
				for (int n = 0; n < iSubsetW; n++)
				{
					//-! Calculate the location of warped subvolume T
					fWarpX =
						POI_.X() + 1 + m_fWarp[0][0] * (n - iSubsetX) + m_fWarp[0][1] * (m - iSubsetY) + m_fWarp[0][2] * (l - iSubsetZ) + m_fWarp[0][3];
					fWarpY =
						POI_.Y() + 1 + m_fWarp[1][0] * (n - iSubsetX) + m_fWarp[1][1] * (m - iSubsetY) + m_fWarp[1][2] * (l - iSubsetZ) + m_fWarp[1][3];
					fWarpZ =
						POI_.Z() + 1 + m_fWarp[2][0] * (n - iSubsetX) + m_fWarp[2][1] * (m - iSubsetY) + m_fWarp[2][2] * (l - iSubsetZ) + m_fWarp[2][3];
					iTempX = int(fWarpX);
					iTempY = int(fWarpY);
					iTempZ = int(fWarpZ);

					m_fTempX = fWarpX - iTempX;
					m_fTempY = fWarpY - iTempY;
					m_fTempZ = fWarpZ - iTempZ;
					if ((iTempX >= 1) && (iTempY >= 1) && (iTempZ >= 1) &&
						(iTempX <= (iOriginVolWidth - 3)) && (iTempY <= (iOriginVolHeight - 3)) && (iTempZ <= (iOriginVolDepth - 3)))
					{
						//-! In most case, it is sub-pixel location, estimate the gary intensity using interpolation
						m_fSubsetT_id_lm[n] = 0;

						w_x[0] = w0_c(m_fTempX);
						w_x[1] = w1_c(m_fTempX);
						w_x[2] = w2_c(m_fTempX);
						w_x[3] = w3_c(m_fTempX);

						w_y[0] = w0_c(m_fTempY);
						w_y[1] = w1_c(m_fTempY);
						w_y[2] = w2_c(m_fTempY);
						w_y[3] = w3_c(m_fTempY);

						w_z[0] = w0_c(m_fTempZ);
						w_z[1] = w1_c(m_fTempZ);
						w_z[2] = w2_c(m_fTempZ);
						w_z[3] = w3_c(m_fTempZ);

						//z
						for (int j = 0; j < 4; ++j) {
							//y
							float **m_fTGBspline_j = m_fTGBspline[iTempZ + 2 - 1 + j];
							for (int i = 0; i < 4; ++i) {
								float *m_fTGBspline_ji = m_fTGBspline_j[iTempY + 2 - 1 + i];
								/*sum_x[i] =
									w_x[0] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 - 1]
									+ w_x[1] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1]
									+ w_x[2] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 1]
									+ w_x[3] * m_fTGBspline[iTempZ + 1 - 1 + j][iTempY + 1 - 1 + i][iTempX + 1 + 2];*/
								sum_x[i] =
									w_x[0] * m_fTGBspline_ji[iTempX + 2 - 1]
									+ w_x[1] * m_fTGBspline_ji[iTempX + 2]
									+ w_x[2] * m_fTGBspline_ji[iTempX + 2 + 1]
									+ w_x[3] * m_fTGBspline_ji[iTempX + 2 + 2];
							}
							sum_y[j] =
								w_y[0] * sum_x[0]
								+ w_y[1] * sum_x[1]
								+ w_y[2] * sum_x[2]
								+ w_y[3] * sum_x[3];
						}

						m_fSubsetT_id_lm[n] =
							w_z[0] * sum_y[0]
							+ w_z[1] * sum_y[1]
							+ w_z[2] * sum_y[2]
							+ w_z[3] * sum_y[3];

#ifdef _JR_DEBUG
						if (l == 0 && m == 16 && n == 0) {
							printf("id:%d; PRef:(%d,%d,%d); Ptar:(%f,%f,%f); val:%f; iTempX:(%d,%d,%d); inerp:(%f)\n", ((l*iSubsetH + m)*iSubsetW + n), n, m, l, fWarpX, fWarpY, fWarpZ, m_fSubsetT_id_lm[n], iTempX, iTempY, iTempZ, m_fTGBspline[iTempZ + 2][iTempY + 2][iTempX + 2]);
						}
#endif

#ifdef _JR_DEBUG
						if (((l*iSubsetH + m)*iSubsetW + n) == 0) {
							printf("id:%d; PRef:(%d,%d,%d); Ptar:(%f,%f,%f); val:%f; iTempX:(%d,%d,%d); inerp:(%f)\n", ((l*iSubsetH + m)*iSubsetW + n), n, m, l, fWarpX, fWarpY, fWarpZ, m_fSubsetT_id_lm[n], iTempX, iTempY, iTempZ, m_fTGBspline[iTempZ + 2][iTempY + 2][iTempX + 2]);
						}
#endif

						fSubAveT += m_fSubsetT_id_lm[n];
					}
					else
					{
						//-! if the loacation of the warped subvolume T is out of the ROI, stop iteration and set p as the current value
						POI_.SetOutofROI(1);
						POI_.SetProcessed(-2);
						return;
					}
				}
			}
		}
		fSubAveT /= fSubsetSize;
		//-! Check if Subset T is a all dark subvolume
		if (fSubAveT == 0)
		{
			POI_.SetDarkSubset(4);
			POI_.SetProcessed(-1);
			return;
		}
		for (int l = 0; l < iSubsetD; l++)
		{
			for (int m = 0; m < iSubsetH; m++)
			{
				for (int n = 0; n < iSubsetW; n++)
				{
					m_fSubsetT_id[l][m][n] = m_fSubsetT_id[l][m][n] - fSubAveT;	// T_i - T_m
					fSubNorT += pow(m_fSubsetT_id[l][m][n], 2);				// Sigma(T_i - T_m)^2
				}
			}
		}
		fSubNorT = sqrt(fSubNorT);	// sqrt (Sigma(T_i - T_m)^2)
		if (fSubNorT == 0)
		{
			POI_.SetDarkSubset(5);	// set flag
			POI_.SetProcessed(-1);
			return;
		}


#ifdef _JR_DEBUG
		printf("iteration%d, normT:%f, normR/normT:%f\n", POI_.Iteration() - 1, fSubNorT, fSubNorR / fSubNorT);
#endif

		//-! Compute the error image
		for (int k = 0; k < 12; k++)
		{
			m_fNumerator[k] = 0;
		}
		fZNCC = 0;
		for (int l = 0; l < iSubsetD; l++)
		{
			for (int m = 0; m < iSubsetH; m++)
			{
				float *m_fSubsetT_id_lm = m_fSubsetT_id[l][m];
				float *m_fSubsetR_id_lm = m_fSubsetR_id[l][m];
				for (int n = 0; n < iSubsetW; n++)
				{
					//fErrors = (fSubNorR / fSubNorT) * m_fSubsetT_id[l][m][n] - m_fSubsetR_id[l][m][n];
					fErrors = (fSubNorR / fSubNorT) * m_fSubsetT_id_lm[n] - m_fSubsetR_id_lm[n];
					fZNCC += m_fSubsetT_id[l][m][n] * m_fSubsetR_id[l][m][n];
					//-! Compute the numerator
					float *m_fRDescent_id_lmn = m_fRDescent_id[l][m][n];
					for (int k = 0; k < 12; k++)
					{
						//m_fNumerator[k] += (m_fRDescent_id[l][m][n][k] * fErrors);
						m_fNumerator[k] += (m_fRDescent_id_lmn[k] * fErrors);
					}
				}
			}
		}
		fZNCC = fZNCC / (fSubNorR*fSubNorT);
		POI_.SetZNCC(fZNCC);
		//-! Compute DeltaP
		for (int k = 0; k < 12; k++)
		{
			DP[k] = 0;
			for (int n = 0; n < 12; n++)
			{
				DP[k] += (m_fInvHessian[k][n] * m_fNumerator[n]);
			}
		}
		POI_.SetDP(DP);
		fDU = POI_.DP()[0];
		fDUx = POI_.DP()[1];
		fDUy = POI_.DP()[2];
		fDUz = POI_.DP()[3];
		fDV = POI_.DP()[4];
		fDVx = POI_.DP()[5];
		fDVy = POI_.DP()[6];
		fDVz = POI_.DP()[7];
		fDW = POI_.DP()[8];
		fDWx = POI_.DP()[9];
		fDWy = POI_.DP()[10];
		fDWz = POI_.DP()[11];

		//-! Update the warp
		//-! m_fTemp: store the denominator of the Inverse W
		m_fTemp =
			((fDWz + 1)*fDVy + (-fDWy * fDVz + (fDWz + 1)))*fDUx +
			(((-fDWz - 1)*fDVx + fDWx * fDVz)*fDUy + ((fDWy*fDVx + (-fDWx * fDVy - fDWx))*fDUz +
			((fDWz + 1)*fDVy + (-fDWy * fDVz + (fDWz + 1)))));
		if (m_fTemp == 0)
		{
			POI_.SetInvertibleHessian(2);
			POI_.SetProcessed(-1);
			return;
		}

		//-! W(P) <- W(P) o W(DP)^-1
		f00 = (fDWz + 1)*fDVy + (-fDWy * fDVz + (fDWz + 1));
		f10 = (-fDWz - 1)*fDVx + fDWx * fDVz;
		f20 = fDWy * fDVx + (-fDWx * fDVy - fDWx);
		f01 = (-fDWz - 1)*fDUy + fDWy * fDUz;
		f11 = (fDWz + 1)*fDUx + (-fDWx * fDUz + (fDWz + 1));
		f21 = -fDWy * fDUx + (fDWx*fDUy - fDWy);
		f02 = fDVz * fDUy + (-fDVy - 1)*fDUz;
		f12 = -fDVz * fDUx + (fDVx*fDUz - fDVz);
		f22 = (fDVy + 1)*fDUx + (-fDVx * fDUy + (fDVy + 1));
		f03 = (-fDW * fDVz + (fDWz + 1)*fDV)*fDUy + ((fDW*fDVy + (-fDWy * fDV + fDW))*fDUz + ((-fDWz - 1)*fDVy + (fDWy*fDVz + (-fDWz - 1)))*fDU);
		f13 = (fDW*fDVz + (-fDWz - 1)*fDV)*fDUx + ((-fDW * fDVx + fDWx * fDV)*fDUz + (((fDWz + 1)*fDVx - fDWx * fDVz)*fDU + (fDW*fDVz + (-fDWz - 1)*fDV)));
		f23 = (-fDW * fDVy + (fDWy*fDV - fDW))*fDUx + ((fDW*fDVx - fDWx * fDV)*fDUy + ((-fDWy * fDVx + (fDWx*fDVy + fDWx))*fDU + (-fDW * fDVy + (fDWy*fDV - fDW))));
		m_fWarp[0][0] = ((1 + fUx)*f00 + fUy * f10 + fUz * f20) / m_fTemp;	m_fWarp[0][1] = ((1 + fUx)*f01 + fUy * f11 + fUz * f21) / m_fTemp; 	m_fWarp[0][2] = ((1 + fUx)*f02 + fUy * f12 + fUz * f22) / m_fTemp; 	m_fWarp[0][3] = ((1 + fUx)*f03 + fUy * f13 + fUz * f23) / m_fTemp + fU;
		m_fWarp[1][0] = (fVx*f00 + (1 + fVy)*f10 + fVz * f20) / m_fTemp;	m_fWarp[1][1] = (fVx*f01 + (1 + fVy)*f11 + fVz * f21) / m_fTemp;	m_fWarp[1][2] = (fVx*f02 + (1 + fVy)*f12 + fVz * f22) / m_fTemp;	m_fWarp[1][3] = (fVx*f03 + (1 + fVy)*f13 + fVz * f23) / m_fTemp + fV;
		m_fWarp[2][0] = (fWx*f00 + fWy * f10 + (1 + fWz)*f20) / m_fTemp;	m_fWarp[2][1] = (fWx*f01 + fWy * f11 + (1 + fWz)*f21) / m_fTemp;	m_fWarp[2][2] = (fWx*f02 + fWy * f12 + (1 + fWz)*f22) / m_fTemp;	m_fWarp[2][3] = (fWx*f03 + fWy * f13 + (1 + fWz)*f23) / m_fTemp + fW;
		m_fWarp[3][0] = 0;	m_fWarp[3][1] = 0;	m_fWarp[3][2] = 0;	m_fWarp[3][3] = 1;

		//-! Update P
		P[0] = m_fWarp[0][3];	P[1] = m_fWarp[0][0] - 1;	P[2] = m_fWarp[0][1];		P[3] = m_fWarp[0][2];
		P[4] = m_fWarp[1][3];	P[5] = m_fWarp[1][0];		P[6] = m_fWarp[1][1] - 1;	P[7] = m_fWarp[1][2];
		P[8] = m_fWarp[2][3];	P[9] = m_fWarp[2][0];		P[10] = m_fWarp[2][1];		P[11] = m_fWarp[2][2] - 1;
		POI_.SetP(P);
		fU = POI_.P()[0];
		fUx = POI_.P()[1];
		fUy = POI_.P()[2];
		fUz = POI_.P()[3];
		fV = POI_.P()[4];
		fVx = POI_.P()[5];
		fVy = POI_.P()[6];
		fVz = POI_.P()[7];
		fW = POI_.P()[8];
		fWx = POI_.P()[9];
		fWy = POI_.P()[10];
		fWz = POI_.P()[11];

#ifdef _JR_DEBUG
		printf("iteration: %d, deltaP:%f\n", POI_.Iteration() - 1, sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)));
		printf("\t\tmatrix:%f %f %f %f, %f %f %f %f, %f %f %f %f \n",
			P[0], P[1], P[2], P[3],
			P[4], P[5], P[6], P[7],
			P[8], P[9], P[10], P[11]);
#endif
	}


	if (sqrt(pow(POI_.DP()[0], 2) + pow(POI_.DP()[4], 2) + pow(POI_.DP()[8], 2)) <= fDeltaP)
		POI_.SetConverge(1);

	POI_.SetProcessed(1);
	return;

}

int icgnRegistration::InverseHessian_GaussianJordan(
	vector<vector<float>>&m_fInvHessian,
	vector<vector<float>>&m_fHessian)
{
	int k, l, m, n;
	int iTemp;
	float dTemp;

	for (l = 0; l < 12; l++)
	{
		for (m = 0; m < 12; m++)
		{
			if (l == m)
				m_fInvHessian[l][m] = 1;
			else
				m_fInvHessian[l][m] = 0;
		}
	}

	for (l = 0; l < 12; l++)
	{
		//-! Find pivot (maximum lth column element) in the rest (6-l) rows
		iTemp = l;
		for (m = l + 1; m < 12; m++)
		{
			if (m_fHessian[m][l] > m_fHessian[iTemp][l])
			{
				iTemp = m;
			}
		}
		if (fabs(m_fHessian[iTemp][l]) == 0)
		{
			return 1;
		}
		//-! Swap the row which has maximum lth column element
		if (iTemp != l)
		{
			for (k = 0; k < 12; k++)
			{
				dTemp = m_fHessian[l][k];
				m_fHessian[l][k] = m_fHessian[iTemp][k];
				m_fHessian[iTemp][k] = dTemp;

				dTemp = m_fInvHessian[l][k];
				m_fInvHessian[l][k] = m_fInvHessian[iTemp][k];
				m_fInvHessian[iTemp][k] = dTemp;
			}
		}
		//-! Perform row operation to form required identity matrix out of the Hessian matrix
		for (m = 0; m < 12; m++)
		{
			dTemp = m_fHessian[m][l];
			if (m != l)
			{
				for (n = 0; n < 12; n++)
				{
					m_fInvHessian[m][n] -= m_fInvHessian[l][n] * dTemp / m_fHessian[l][l];
					m_fHessian[m][n] -= m_fHessian[l][n] * dTemp / m_fHessian[l][l];
				}
			}
			else
			{
				for (n = 0; n < 12; n++)
				{
					m_fInvHessian[m][n] /= dTemp;
					m_fHessian[m][n] /= dTemp;
				}
			}
		}
	}
	return 0;
}
