#ifndef __PARAMETER_H_
#define __PARAMETER_H_


typedef double typeRate;

extern int MAX_ITER;			// 最大迭代次数
extern double lambda;			// 正则化系数
extern double gamma;			// 学习率


extern typeRate **matrixRate;	// 评分矩阵 size: M * N
extern typeRate *matrixUser;	// size: K * M
extern typeRate *matrixItem;	// size: K * N
extern int M;	// matrixRate 行数
extern int N;	// matrixRate 列数
extern int K;	// 隐含向量维数
extern int subBlockNumL;	// subBlockNumL * subBlockNumL个子块
extern int subBlockNum;	// 子块总数目
extern int subBlockLen;	// 子块大小为 subBlockLen * subBlockLen
extern int subBlockNodeNum; // 子块大小size（包含0与非0）



#endif