#ifndef __PIDVC_COMPUTE_H__
#define __PIDVC_COMPUTE_H__

#include "POI.h"

class computePOI {

protected:
	int thread_num = 1;

	//subset radius
	int m_iSubsetX = 0;
	int m_iSubsetY = 0;
	int m_iSubsetZ = 0;

public:
	void setSubsetRadius(const int SubsetX, const int SubsetY, const int SubsetZ) {
		m_iSubsetX = SubsetX;
		m_iSubsetY = SubsetY;
		m_iSubsetZ = SubsetZ;
	};

	virtual void preCompute() = 0;
	virtual void free() = 0;
	virtual void compute(CPOI &POI_) = 0;
};

#endif