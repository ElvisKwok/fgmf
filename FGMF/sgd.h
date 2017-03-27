#ifndef __GSGD_H_
#define __GSGD_H_

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

#include <cuda_runtime.h>
#include <device_launch_parameters.h>   //  vs消除未定义的提示
#include <device_functions.h>
#include <cublas_v2.h>

#include <iostream>
#include <algorithm>

#include "fgmf.h"
#include "parameter.h"

using namespace std;

extern int threads_per_block;

void solveByGPU(
    sRateNode *rateNodeArray,
    typeRate *matrixUser,
    typeRate *matrixItem,
    sWorkset *worksetArray,
    sWorkseg *mWorkseg,
    int *matrixPattern,
    int subBlockNumL,           // subBlockNumL * subBlockNumL个子块
    int subBlockLen,            // 子块大小为 subBlockLen * subBlockLen
    double lambda,              // 正则化系数
    double gamma,               // 学习率
    int NNZ                     // 评价值个数
);

void solveByGPU(
	sMicroNode *microNodeArray,
	typeRate *matrixUser,
	typeRate *matrixItem,
	sWorkset *worksetArray,
	sWorkseg *mWorkseg,
	int *matrixPattern,
	int subBlockNumL,           // subBlockNumL * subBlockNumL个子块
	int subBlockLen,            // 子块大小为 subBlockLen * subBlockLen
	double lambda,              // 正则化系数
	double gamma,               // 学习率
	int NNZ                     // 评价值个数
);

#endif