#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <iostream>

#include "fgmf.h"
#include "parameter.h"

using namespace std;


void callKernel(sRateNode *rateNodeArray, typeRate *matrixP, typeRate *matrixQ, int M, int N, int K, sWorkset *worksetArray, sWorkseg *mWorkseg, int *mPattern, int subBlockNumL, int subBlockLen, int NNZ);