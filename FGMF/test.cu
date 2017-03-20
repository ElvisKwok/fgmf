#include "test.h"

#define lambda 0.05
#define gamma 0.003

typeRate computeRMSE_new(sRateNode *rateNodeArray, typeRate *matrixP, typeRate *matrixQ, int K, int NNZ)
{
	typeRate rmse = 0;
	for (int i = 0; i < NNZ; ++i)
	{
		typeRate rate = rateNodeArray[i].rate;
		int userIdx = rateNodeArray[i].u-1;
		int itemIdx = rateNodeArray[i].i-1;
		typeRate predictRate = 0;
		for (int k = 0; k < K; ++k)
		{
			predictRate += (*(matrixP + userIdx * K + k)) * (*(matrixQ + itemIdx * K + k));
		}
		rmse += pow((rate - predictRate), 2);
	}

	return sqrt(rmse / NNZ);
}

__device__ void sgdUpdateNew(typeRate rate, typeRate *matrixP, typeRate *matrixQ, int userIdx, int itemIdx, int K)
{
	typeRate predictRate = 0;
	for (int k = 0; k < K; ++k)
	{
		predictRate += (*(matrixP + userIdx * K + k)) * (*(matrixQ + itemIdx * K + k));
	}
	typeRate err = rate - predictRate;
	for (int k = 0; k < K; ++k)
	{
		(*(matrixP + userIdx * K + k)) += gamma * (2 * err * (*(matrixQ + itemIdx * K + k)) - lambda * (*(matrixP + userIdx * K + k)));
		(*(matrixQ + itemIdx * K + k)) += gamma * (2 * err * (*(matrixP + userIdx * K + k)) - lambda * (*(matrixQ + itemIdx * K + k)));
	}
}

__global__ void sgd_kernelNew(sRateNode *d_rateNodeArray, typeRate *d_matrixP, typeRate *d_matrixQ, int K,
	sWorkset *d_worksetArray, sWorkseg *d_mWorkseg, int *d_mPattern, int s,
	int subBlockNumL, int subBlockLen)
{
	int tid = threadIdx.x;
	int tbid = blockIdx.x;
	if (tbid > subBlockNumL)
		return;
	int bid = *(d_mPattern + s * subBlockNumL + tbid);
	if (bid == -1 || d_worksetArray[bid].beg == d_worksetArray[bid].end)
		return;

	for (int tag = 0; tag < subBlockLen; ++tag)
	{
		int from = (*(d_mWorkseg + tbid*subBlockLen + tag)).from;
		int to = (*(d_mWorkseg + tbid*subBlockLen + tag)).to;
		for (int iRate = from + tid; iRate < to; iRate += blockDim.x)
		{
			typeRate rate = d_rateNodeArray[iRate].rate;
			int userIdx = d_rateNodeArray[iRate].u-1;
			int itemIdx = d_rateNodeArray[iRate].i-1;
			sgdUpdateNew(rate, d_matrixP, d_matrixQ, userIdx, itemIdx, K);
		}
		__syncthreads();
	}
}


void callKernel(sRateNode *rateNodeArray, typeRate *matrixP, typeRate *matrixQ, int M, int N, int K,
	sWorkset *worksetArray, sWorkseg *mWorkseg, int *mPattern,
	int subBlockNumL, int subBlockLen, int NNZ)
{
	sRateNode *d_rateNodeArray;
	typeRate *d_matrixP;
	typeRate *d_matrixQ;
	sWorkset *d_worksetArray;
	sWorkseg *d_mWorkseg;
	int *d_mPattern;

	int subBlockNum = subBlockNumL * subBlockNumL;

	cudaMalloc((void**)&d_rateNodeArray, NNZ * sizeof(sRateNode));
	cudaMalloc((void**)&d_matrixP, M*K * sizeof(typeRate));
	cudaMalloc((void**)&d_matrixQ, N*K * sizeof(typeRate));
	cudaMalloc((void**)&d_worksetArray, subBlockNum * sizeof(sWorkset));
	cudaMalloc((void**)&d_mWorkseg, subBlockNum*subBlockLen * sizeof(sWorkseg));
	cudaMalloc((void**)&d_mPattern, subBlockNum * sizeof(int));

	cudaMemcpy(d_rateNodeArray, rateNodeArray, NNZ * sizeof(sRateNode), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixP, matrixP, M*K * sizeof(typeRate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixQ, matrixQ, N*K * sizeof(typeRate), cudaMemcpyHostToDevice);
	cudaMemcpy(d_worksetArray, worksetArray, subBlockNum * sizeof(sWorkset), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mWorkseg, mWorkseg, subBlockNum*subBlockLen * sizeof(sWorkseg), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mPattern, mPattern, subBlockNum * sizeof(int), cudaMemcpyHostToDevice);

	for (int iter = 0; iter < MAX_ITER; ++iter)
	{
		auto start = system_clock::now();
		for (int s = 0; s < subBlockNumL; ++s)
		{
			sgd_kernelNew << < subBlockNumL, 1024 >> >(
				d_rateNodeArray,
				d_matrixP,
				d_matrixQ,
				K,
				d_worksetArray,
				d_mWorkseg,
				d_mPattern,
				s,
				subBlockNumL,
				subBlockLen
				);
			cudaThreadSynchronize();
		}
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		cout << "it takes iter " << iter << "\t\t" << double(duration.count()) * microseconds::period::num / microseconds::period::den << " seconds" << endl;
		cudaMemcpy(matrixP, d_matrixP, M*K * sizeof(typeRate), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrixQ, d_matrixQ, N*K * sizeof(typeRate), cudaMemcpyDeviceToHost);
		cout << "RMSE: " << computeRMSE_new(rateNodeArray, matrixP, matrixQ, K, NNZ) << endl;
	}

	cudaFree(d_rateNodeArray);
	cudaFree(d_matrixP);
	cudaFree(d_matrixQ);
	cudaFree(d_worksetArray);
	cudaFree(d_mWorkseg);
	cudaFree(d_mPattern);
}
