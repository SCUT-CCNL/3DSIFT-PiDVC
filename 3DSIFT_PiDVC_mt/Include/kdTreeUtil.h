#ifndef __PIDVC_KD_TREE_UTIL_H__
#define __PIDVC_KD_TREE_UTIL_H__

#include <kdtree.h>
#include <vector>
//#include 

// -! KD-Tree Tools
void KD_Build(kdtree *&_kd, int *&_idx, const std::vector<CPUSIFT::Cvec> &vc);
void KD_Destroy(kdtree *&_kd, int *&_idx);
void KD_RangeSerach(kdtree* _kd, std::vector<CPUSIFT::Cvec> &neighbor, std::vector<int> &idx, const CPUSIFT::Cvec query, const double range);
void KD_KNNSerach(kdtree* _kd, std::vector<CPUSIFT::Cvec> &neighbor, std::vector<int> &idx, const CPUSIFT::Cvec query, const int K);

#endif