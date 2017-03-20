#ifndef __GPUSGD_SERIAL_H_
#define __GPUSGD_SERIAL_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>   
#include <map>

#include "basic_func.h"
#include "parameter.h"
#include "parallelReadFile.h"
using namespace std;
using std::cout;

using namespace chrono;

// CALL_FUN_TIME(test(10))
#define CALL_FUN_TIME(FUN) \
{ \
	auto start = system_clock::now(); \
	(FUN); \
	auto end = system_clock::now(); \
	auto duration = duration_cast<microseconds>(end - start); \
	cout << "it takes " #FUN ": \t\t" << double(duration.count()) * microseconds::period::num / microseconds::period::den << " seconds" << endl; \
}

// debug
template <typename T>
void printVar(string varName, const T &var)
{
	cout << varName << ": \t" << var << endl;
}
inline void printLine() { cout << "#######################################" << endl; }
void unitTest();

// 读取输入文件：
// M, N
// (user, item, rate)
void readFile(string fileName = "input.txt");

// 输出训练模型
void writeFile(string fileName = "model.txt");

// 输出预测结果
// resultArray按user_item的输出顺序
void writeFile(typeRate *result, int predictNum, string fileName = "predict_result.txt");

// 初始化参数
void initParameter();

// 根据输入的M, N 初试化subBlock维度参数
void initBlockDimension();

// 初始化所有数据（指针）并分配空间，调用newArray, newMatrix
void initAllData();


// sWorkset: 记录每个子块b_xy（bid）在评分矩阵R的边界beg, end
struct sWorkset{
	int beg;
	int end;

	// 无用的成员函数
	/*
	sWorkset() : beg(0), end(0){}
	sWorkset(int a, int b) : beg(a), end(b) {}
	sWorkset &operator=(const sWorkset &rhs)
	{
	// 首先检测等号右边的是否就是左边的对象本，若是本对象本身,则直接返回
	if (this == &rhs)
	{
	return *this;
	}

	// 复制等号右边的成员到左边的对象中
	this->beg = rhs.beg;
	this->end = rhs.end;

	// 把等号左边的对象再次传出
	// 目的是为了支持连等 eg:    a=b=c 系统首先运行 b=c
	// 然后运行 a= ( b=c的返回值,这里应该是复制c值后的b对象)
	return *this;
	}
	void print(){ cout << beg << "\t" << end << endl; }
	void setBeg(int subBlockIdxX, int subBlockIdxY);
	void setEnd(int subBlockIdxX, int subBlockIdxY);

	void setFromInput(int a, int b) { beg = a; end = b; }
	*/
};


// sWorkseg: 记录子块bid中每个评价值组tag的边界from和to
struct sWorkseg {
	int from;
	int to;
};


// 评分矩阵非零元素
struct sRateNode
{
	int u;	// userIdx
	int i;	// itemIdx
	typeRate rate;
	int bid;
	int subBlockIdxX;
	int subBlockIdxY;
	int label;

	sRateNode(){}
	sRateNode(int inputU, int inputI, typeRate inputRate) :
		u(inputU), i(inputI), rate(inputRate) {}
};

// 计算rateNode所属的子块下标x y和bid
void setSubBlockIdx(sRateNode &node);

// 计算rateNode的label
void setLabel(sRateNode &node);

// 矩阵变换后要重新计算bid(变换中)
void resetAllNode_blockIdx();

// 矩阵变换后要重新计算bid和label（final）
void resetAllNode_blockIdx_label();

// sort比较谓语pred: 先按bid排序，bid相同按label排序
bool compare_bid_label(sRateNode a, sRateNode b);

// 根据bid将rateNodeArray数组排序
void sortRateNodeArrayBid();

// block子方块
struct sSubBlock
{
	int bid;
	int subBlockIdxX;
	int subBlockIdxY;
	int rateNum;
	sRateNode *subBlockNodeArray;
	int pattern;

	sSubBlock() {}
	sSubBlock(int blockId, int size, sRateNode *nodeArray = NULL) :
		bid(blockId), rateNum(size), subBlockNodeArray(nodeArray){}
};

// DELETE
// setSubBlockIdx()
// setRateNum() 
// allocSubBlockNodeArray() 
// labelNodeInSubBlock()
/*
// 根据bid设置子块x y坐标
void setSubBlockIdx(sSubBlock &subBlock);

// 设置子块包含的非0元素个数
void setRateNum(sSubBlock &subBlock);


// 取消：直接复制指针，不要重复分配空间
// DELETE
// 分配子块的node数组空间
//void allocSubBlockNodeArray(sSubBlock &subBlock);


// 对子块内所有rateNode进行label
void labelNodeInSubBlock(sSubBlock &subBlock);
*/




// DELETE
// workseg计算
/*
// 计算子块bid中所有标签的评价值数目，保存到seg数组的第bid行
void computeSeg(int bid);

// 利用记录的数组seg得出每个workseg的from和to
// 计算workseg(bid, label)的from和to，即子块bid中评价值label的起始位置
void computeWorkseg(int bid, int tag);
*/

// 计算所有子块的pattern到二维数组pattern
void setAllPattern();

// DELETE
// 计算pattern
/*
// 设置子块所属的pattern
void setSubBlockPattern(sSubBlock &subBlock);

// 计算所有子块的pattern到二维数组pattern
void setAllPattern();
*/


// DELETE
// 子块内rateNode排序
/*
// sort比较谓语pred
bool compare_label(sRateNode a, sRateNode b);

// 根据label对子块内所有rateNode进行排序
void sortLabelInSubBlock(sSubBlock &subBlock);

// 调用sortLabelInSubBlock，对所有子块进行label排序
void sortLabelAll();
*/

// 取消：直接复制指针，不要重复分配空间
// DELETE
/*
// 释放子块的动态内存空间
void destroySubBlock(sSubBlock &subBlock);
*/

// DELETE
// 利用记录的数组matrixSubset得出单个workset的beg和end
/*
void setWorkset(int bid);
*/

// 扫描bid排序后的rateNodeArray, 得出每个workset的beg和end，保存到worksetArray
// 同时统计每个subset的值,保存到subsetArray
void setWorkset();

// 扫描bid_label排序后的rateNodeArray, 得出每个workseg的from和to，保存到mWorkseg
void setWorkseg();


// 数组array内存分配, 并初始化为val
template <typename T>
void newArray(T * &array, int n, int val);

// DELETE:
// 矩阵m内存分配, 并初始化为val
template <typename T>
void newMatrix(T ** &m, int rowNum, int colNum, int val);

// 1维的矩阵m内存分配, 并随机初始化
template <typename T>
void newMatrixRandom1D(T* &m, int rowNum, int colNum);

// 矩阵m内存销毁
void deleteMatrix(typeRate **m, int rowNum, int colNum);

// 随机初始化矩阵
void randomInitMatrix(typeRate **m, int rowNum, int colNum);



// 输出子块nnz的max_diff
int getMaxDiff();

// 输出子块均匀度
double getEvenness();

// 在乱序的rateNodeArray计算subsetArray，省下排序时间
void computeSubsetArray();

// 矩阵行shuffle NNZ版本(基于任意顺序的rateNodeArray)
// note: 要保存bestPermRow，用于复原, permRow大小为M, 初始化为{1, ..., M}
void rowShuffle(vector<int> &curPermRow, vector<int> &bestPermRow);

// 矩阵列shuffle NNZ版本(基于任意顺序的rateNodeArray)
// note: 要保存bestPermColumn，用于复原, permColumn大小为M, 初始化为{1, ..., N}
void columnShuffle(vector<int> &curPermColumn, vector<int> &bestPermColumn);

// 矩阵随机变换(最优解)
void matrixShuffle();

// TO-DO
// rateNodeArray指向备份矩阵，应用最终的变换
void matrixShuffleApply();

// TO-DO
// 矩阵变换后，还原matrixUser, matrixItem到正确的行、列，用于预测。
void matrixShuffleRecover();

// DELETE: 矩阵shuffle原始版(随机生成矩阵的函数可以用下)
/*
// 矩阵的行/列 shuffle (调用rowShuffle)
void randomShuffleMatrix(typeRate **m, int rowNum, int columnNum);

// 矩阵分块，并统计每块包含的元素个数
int blockMatrix(typeRate **a, int rowNums, int blockLen);

// test shuffle
void randomGenerateMatrix();
*/


// DELETE
// 检查子块bid是否越界
/*
// 返回值：
// 0 不越界
// 1 完全越界
// 2 部分越界
int checkSubBlockBoundary(int bid);
*/

// 计算子块b_xy的ID，其中子块大小为: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockNumL, int x, int y);

// 计算bid对应的子块x,y下标
void getBlockXY(int bid, int &x, int &y);

// 计算Rui所属的子块x,y下标
void getBlockXY(int u, int i, int &x, int &y);


// 记录子块b_xy的ID: bid到二维数组pattern(s, t) 
// 子块b_xy 是第s种模式中的第t个子块, 则把computeSubBlockID(z, x, y)的结果放入pattern(s,t)
void setPattern(int s, int t, int x, int y);

// 记录子块b_xy的ID: bid到二维数组pattern(s, t) 
// 子块bid是第s种模式中的第t个子块, 则把bid放入pattern(s,t)
void setPattern(int s, int t, int bid);

// DELETE
// 原subset矩阵的计算
/*
// 返回：subset(x, y)
// 子块 b_xy 包含的评价值个数(非零元素)
int computeSubset(int subBlockIdxX, int subBlockIdxY);

// 记录所有subset(x, y)到全局二维数组matrixSubset
// 子块 b_xy 包含的评价值个数(非零元素)
void computeAllSubset();
*/



// 总执行函数
void execute();

// sgd CPU update
void sgd_CPU();
// FGMF CPU版本
void FGMF_CPU();


#endif