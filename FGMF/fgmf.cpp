#include "fgmf.h"
#include "sgd.h"
#include "test.h"

string parameterFile = "input/parameter.txt";
string inputFileTest = "input/ra.test_awk";
string inputFile = "input/ra.train_awk";
//string inputFile = "input/ra.test_awk";
//string inputFile = "input/BDMovie";
//string inputFile = "input/input.txt";
//string inputFile = "input/10by10.txt";
string resultFile = "output/predict_result.txt";
string modelFile = "output/model.txt";


// 各种模块测试选项
int experimentVar1;			// 输入变化的命令行参数
int testReadFile = 0;		// 只测试读文件？
int testShuffle = 0;		// 只测试读文件？

int MAX_ITER;               // 最大迭代次数
int MAX_SHUFFLE;            // 最大矩阵变换次数
double lambda;              // 正则化系数
double gamma;               // 学习率
int K;                      // 隐含向量维数
int subBlockNumL = 64;      // subBlockNumL * subBlockNumL个子块
int threads_per_block;      // 一个block包含的线程数
int ompNumThread;           // 限制omp线程数，用于实验

int isShuffle;

// DELETE
typeRate **matrixRate;      // 评分矩阵 size: M * N

typeRate *matrixUser;       // size: M * K
typeRate *matrixItem;       // size: N * K
int M;                      // matrixRate 行数
int N;                      // matrixRate 列数
int subBlockNum;            // 子块总数目, value = subBlockNumL * subBlockNumL
int subBlockLen;            // 子块大小为 subBlockLen * subBlockLen, value = max(M, N) / subBlockNumL
int subBlockNodeNum;        // 子块大小size（包含0与非0）, value = subBlockLen * subBlockLen

int NNZ;                    // 非零元素个数，输入文件(user, item, rate)的行数
int NNZ_test;

int *matrixPattern;     // size: subBlockNumL * subBlockNumL, 即 模式s * 子块t (注意初始化为-1)
// DELETE
int **matrixSubset;         // size: (subBlockNumL * subBlockNumL)
int *subsetArray;           // 记录每个子块bid的nnz, size: subBlockNum, bid为下标 （用于统计块均与度）

// 子块bid在评分矩阵中的边界beg和end
sWorkset *worksetArray;     // size: subBlockNum

// DELETE
// 子块bid中标有tag标签的评价值数目
int **mSeg;                 // size: subBlockNum * subBlockLen, 即 bid*tag

// 子块bid中每个评价值组tag的边界from和to
sWorkseg *mWorkseg;     // size: subBlockNum * subBlockLen, 即 bid*tag


vector<sRateNode> rateNodeVector;   // 用于动态读取输入
vector<sRateNode> rateNodeVector_backup;    // 原始备份，矩阵变换后的应用
sRateNode *rateNodeArray;           // 随着矩阵变换而改动

vector<sRateNode> rateNodeVectorTest;	// test dataset
sRateNode *rateNodeArrayTest;

// DELETE ?
sSubBlock *subBlockArray;   // size: subBlockNum个子块

vector<int> permRow;        // 可用于应用/还原，保存最终的行变换策略，size: M (注意值的范围是1~M，而不是0~M-1)
vector<int> permColumn;     // 保存最终的列变换策略，size: N

// 暂时不用
// 用于正式测试只有userIdx和itemIdx的数据集
/*
vector<sRateNode> rateNodeVector_predict;   // 用于动态读取输入
sRateNode *rateNodeArray_predict;           // &rateNodeVector_predict[0]
*/

/********************************* BEGIN:  File Process *********************************/
// 读取输入文件：
// M, N
// (user, item, rate)
void readFile(string fileName)
{
    printVar("inputFile", inputFile);
    // serial read
    /*
    ifstream inputFile(fileName);
    int userIdx;
    int itemIdx;
    typeRate rate;
    while (!inputFile.eof())
    {
    inputFile >> userIdx >> itemIdx >> rate;
    sRateNode node(userIdx, itemIdx, rate);
    setSubBlockIdx(node);
    setLabel(node);

    rateNodeVector.push_back(node);
    }
    */

    // parallel read
	parallelReadFile(fileName, rateNodeVector, ompNumThread);
    NNZ = rateNodeVector.size();
    rateNodeArray = &rateNodeVector[0];

	// test data
	parallelReadFile(inputFileTest, rateNodeVectorTest, ompNumThread);
	NNZ_test = rateNodeVectorTest.size();
	rateNodeArrayTest = &rateNodeVectorTest[0];
}


// debug:
void printRateNode(sRateNode &node)
{
    printVar("node.u", node.u);
    printVar("node.i", node.i);
    printVar("node.rate", node.rate);
    printVar("node.bid ", node.bid);
    printVar("node.subBlockIdxX", node.subBlockIdxX);
    printVar("node.subBlockIdxY ", node.subBlockIdxY);
    printVar("node.label", node.label);
}

// debug:
void printMatrixPattern()
{
    cout << "\nprintMatrixPattern: " << endl;

    for(int i = 0; i < subBlockNumL; ++i)
    {
        for(int j = 0; j < subBlockNumL; ++j)
        {
            cout << (matrixPattern + i * subBlockNumL)[j] << " ";
        }

        cout << endl;
    }

    cout << endl;
}

// debug:
void printWorksetArray()
{
    cout << "printWorksetArray: " << endl;

    for(int i = 0; i < subBlockNum; ++i)
    {
        cout << "[" << worksetArray[i].beg << ", " << worksetArray[i].end << ")\t";
    }

    cout << '\n' << endl;
}

// debug:
void printMatrixWorkseg()
{
    cout << "printMatrixWorkseg: " << endl;

    for(int i = 0; i < subBlockNum; ++i)
    {
        for(int j = 0; j < subBlockLen; ++j)
        {
            cout << "[" << (mWorkseg + i * subBlockLen)[j].from << ", " << (mWorkseg + i * subBlockLen)[j].to << ")\t";
        }

        cout << endl;
    }

    cout << endl;
}


// 输出训练模型(matrixUser, matrixItem)
void writeFile(string fileName)
{
    ofstream outFile(fileName);

    for(int i = 0; i < M; ++i)
    {
        for(int k = 0; k < K; ++k)
        {
            outFile << *(matrixUser + i * K + k) << " ";
        }

        outFile << endl;
    }

    outFile << endl;

    for(int i = 0; i < N; ++i)
    {
        for(int k = 0; k < K; ++k)
        {
            outFile << *(matrixItem + i * K + k) << " ";
        }

        outFile << endl;
    }
}

// TO-DO
// 输出预测结果
// resultArray按user_item的输出顺序
void writeFile(typeRate *resultArray, int predictNum, string fileName)
{
    ofstream outFile(fileName);

    for(int i = 0; i < predictNum; ++i)
    {
        outFile << resultArray[i] << endl;
    }
}

// 根据参数文件设定，初始化参数
void initParameter()
{
    cout << "reading parameters from parameters.txt..." << endl;
    freopen(parameterFile.c_str(), "r", stdin);
    scanf("MAX_ITER = %d\n", &MAX_ITER);
    scanf("MAX_SHUFFLE = %d\n", &MAX_SHUFFLE);
    scanf("lambda = %lf\n", &lambda);
    scanf("gamma = %lf\n", &gamma);
    scanf("M = %d\n", &M);
    scanf("N = %d\n", &N);
    scanf("K = %d\n", &K);
    scanf("subBlockNumL = %d\n", &subBlockNumL);
    scanf("threads_per_block = %d\n", &threads_per_block);
    scanf("ompNumThread = %d\n", &ompNumThread);
    scanf("isShuffle = %d\n", &isShuffle);
	scanf("testReadFile = %d\n", &testReadFile);
	scanf("testShuffle = %d\n", &testShuffle);

	// 设置命令行实验参数(默认从文件读固定的)
	//ompNumThread = experimentVar1;
	
    fclose(stdin);
    freopen("CON", "r", stdin);   //"CON"代表控制台
    initBlockDimension();
    ///*
    // test
    cout << "MAX_ITER = " << MAX_ITER << endl;
    cout << "MAX_SHUFFLE = " << MAX_SHUFFLE << endl;
    cout << "lambda = " << lambda << endl;
    cout << "gamma = " << gamma << endl;
    cout << "M = " << M << endl;
    cout << "N = " << N << endl;
    cout << "K = " << K << endl;
    cout << "subBlockNumL = " << subBlockNumL << endl;
    cout << "threads_per_block = " << threads_per_block << endl;
    cout << "ompNumThread = " << ompNumThread << endl;
    cout << "isShuffle = " << isShuffle << endl;
	cout << "testReadFile = " << testReadFile << endl;
	cout << "testShuffle = " << testShuffle << endl;
    cout << endl;
    //*/
    //getchar();
}

// 根据输入的M, N 初试化subBlock维度参数
void initBlockDimension()
{
    subBlockNum = subBlockNumL * subBlockNumL;      // 子块总数目
    int maxDimension = max(M, N);
    subBlockLen = maxDimension / subBlockNumL + (maxDimension % subBlockNumL > 0);  // 子块大小为 subBlockLen * subBlockLen
    subBlockNodeNum = subBlockLen * subBlockLen;    // 子块大小size（包含0与非0）
    // debug:
    printVar("subBlockNum", subBlockNum);
    printVar("subBlockLen", subBlockLen);
    //printVar("subBlockNodeNum", subBlockNodeNum);
    printLine();
}

// 初始化所有数据（指针）并分配空间，调用newArray, newMatrix
void initAllData()
{
    // TO-DO
    // 随机初始化
    //typeRate *matrixUser;     // size: M * K
    //typeRate *matrixItem;     // size: N * K
    newMatrixRandom1D(matrixUser, M, K);
    newMatrixRandom1D(matrixItem, N, K);
    //int *matrixPattern;       // size: subBlockNumL * subBlockNumL, 即 模式s * 子块t (注意初始化为-1)
    newArray(matrixPattern, subBlockNum, -1);
    //int *subsetArray;         // 记录每个子块bid的nnz, size: (subBlockNumL * subBlockNumL), bid为下标 （用于统计块均与度）
    newArray(subsetArray, subBlockNum, 0);
    //sWorkset *worksetArray;       // 子块bid在评分矩阵中的边界beg和end, size: (subBlockNumL * subBlockNumL)
    newArray(worksetArray, subBlockNum, 0);
    //sWorkseg *mWorkseg;       // 子块bid中每个评价值组tag的边界from和to, size: (subBlockNumL * subBlockNumL) * subBlockLen, 即 bid*tag
    newArray(mWorkseg, subBlockNum * subBlockLen, 0);
    // DELETE: 交给matrix_shuffle
    // 初始化permRow, permColumn
    /*
    int i;
    for (i = 0; (i < M) && (i < N); ++i)
    {
    permRow.push_back(i+1);
    permColumn.push_back(i+1);
    }
    while (i < M)
    {
    permRow.push_back(i+1);
    ++i;
    }
    while (i < N)
    {
    permColumn.push_back(i+1);
    ++i;
    }
    */
}

/********************************** END:  File Process **********************************/

/********************************* BEGIN: class sRateNode *********************************/
// 计算rateNode所属的子块下标x y和bid
void setSubBlockIdx(sRateNode &node)
{
    node.subBlockIdxX = (node.u - 1) / subBlockLen;     // userIdx从一开始
    node.subBlockIdxY = (node.i - 1) / subBlockLen;
    node.bid = node.subBlockIdxX * subBlockNumL + node.subBlockIdxY;
}

void setLabel(sRateNode &node)
{
    int u = node.u;
    int i = node.i;
    int subBlockIdxX = node.subBlockIdxX;
    int subBlockIdxY = node.subBlockIdxY;
    int deltaI = u - subBlockIdxX * subBlockLen;
    int deltaJ = i - subBlockIdxY * subBlockLen;
    node.label = (subBlockLen - deltaI + deltaJ) % subBlockLen;
}

// 矩阵变换后要重新计算bid(变换中)
void resetAllNode_blockIdx()
{
    for(int i = 0; i < NNZ; ++i)
    {
        setSubBlockIdx(rateNodeArray[i]);
    }
}

// 矩阵变换后要重新计算bid和label（final）
void resetAllNode_blockIdx_label()
{
    for(int i = 0; i < NNZ; ++i)
    {
        setSubBlockIdx(rateNodeArray[i]);
        setLabel(rateNodeArray[i]);
    }
}


// sort比较谓语pred
bool compare_bid_label(sRateNode a, sRateNode b)
{
    // return a.bid < b.bid; // 只按bid排序
    if(a.bid < b.bid)
    {
        return 1;
    }

    if(a.bid == b.bid && a.label < b.label)
    {
        return 1;
    }

    return 0;
}

// 根据bid将rateNodeArray数组排序
void sortRateNodeArrayBid()
{
    sort(rateNodeArray, rateNodeArray + NNZ, compare_bid_label);
}

// sort比较谓语pred
bool compare_user_item(sRateNode a, sRateNode b)
{
    // return a.bid < b.bid; // 只按bid排序
    if(a.u < b.u)
    {
        return 1;
    }

    if(a.u == b.u && a.i < b.i)
    {
        return 1;
    }

    return 0;
}

// 根据user_item将rateNodeArray数组排序
void sortRateNodeArrayUserItem()
{
    sort(rateNodeArray, rateNodeArray + NNZ, compare_user_item);
}
/********************************** END: class sRateNode **********************************/


/********************************* BEGIN: class sSubBlock *********************************/
// DELETE
// setSubBlockIdx()
// setRateNum()
// allocSubBlockNodeArray()
// labelNodeInSubBlock()
#if 0
// 根据bid设置子块x y坐标
void setSubBlockIdx(sSubBlock &subBlock)
{
    subBlock.subBlockIdxX = (subBlock.bid / subBlockLen) + 1;
    subBlock.subBlockIdxY = subBlock.bid % subBlockLen;
}

// 设置子块包含的非0元素个数
void setRateNum(sSubBlock &subBlock)
{
    subBlock.rateNum = computeSubset(subBlock.subBlockIdxX, subBlock.subBlockIdxY);
}

// DELETE
// 取消：直接复制指针，不要重复分配空间
/*
// 分配子块的node数组空间
void allocSubBlockNodeArray(sSubBlock &subBlock)
{
subBlock.subBlockNodeArray = new sRateNode[subBlock.rateNum];
}
*/

// 对子块内所有rateNode进行label
void labelNodeInSubBlock(sSubBlock &subBlock)
{
    for(int i = 0; i < subBlock.rateNum; ++i)
    {
        setLabel(subBlock.subBlockNodeArray[i]);
    }
}

#endif


// DELETE
// workseg计算
/*
// 计算子块bid中所有label标签的评价值数目seg(bid, label)，保存到mSeg数组的第bid行
void computeSeg(int bid)
{
memset(mSeg[bid], 0, subBlockLen * sizeof(int));
int rateNum = subBlockArray[bid].rateNum;
for (int i = 0; i < rateNum; ++i)
{
int label = subBlockArray[bid].subBlockNodeArray[i].label;
++mSeg[bid][label];
}
}

// 利用记录的数组seg得出每个workseg的from和to
// 计算workseg(bid, label)的from和to，即子块bid中评价值label的起始位置
void computeWorkseg(int bid, int tag)
{
mWorkseg[bid][tag].from = worksetArray[bid].beg;    // 包含tag == 0 的情况
for (int i = 0; i <= tag - 1; ++i)
{
mWorkseg[bid][tag].from += mSeg[bid][i];
}
mWorkseg[bid][tag].to = mWorkseg[bid][tag].from + mSeg[bid][tag];
}
*/

/*
// 设置子块所属的pattern
void setSubBlockPattern(sSubBlock &subBlock)
{
int x = subBlock.subBlockIdxX;
int y = subBlock.subBlockIdxY;
subBlock.pattern = (subBlockNumL - x + y) % subBlockNumL;
}

// 计算所有子块的pattern到二维数组pattern
void setAllPattern()
{
vector<int> cnt(subBlockNumL, 0);
for (int i = 0; i < subBlockNum; ++i)
{
int pattern = subBlockArray[i].pattern;
int t = cnt[pattern]++;     // 第s模式的第t个子块
int bid = subBlockArray[i].bid;
setPattern(pattern, t, bid);
}
}
*/

// 计算所有子块的pattern到二维数组pattern
void setAllPattern()
{
    int x, y, pattern;
    vector<int> cnt(subBlockNumL, 0);

    for(int bid = 0; bid < subBlockNum; ++bid)
    {
        if(subsetArray[bid] > 0)
        {
            getBlockXY(bid, x, y);
            pattern = (subBlockNumL - x + y) % subBlockNumL;
            int t = cnt[pattern]++;     // 第s模式的第t个子块
            setPattern(pattern, t, bid);
            //debug:
            /*
            printVar("pattern", pattern);
            printVar("t", t);
            printVar("bid", bid);
            */
        }
    }
}

// DELETE
// 子块内rateNode排序
/*
bool compare_label(sRateNode a, sRateNode b)
{
return a.label < b.label;
}

// 根据label对子块内所有rateNode进行排序
void sortLabelInSubBlock(sSubBlock &subBlock)
{
sort(subBlock.subBlockNodeArray, subBlock.subBlockNodeArray + subBlock.rateNum, compare_label);
}

// 调用sortLabelInSubBlock，对所有子块进行label排序
void sortLabelAll()
{
for (int i = 0; i < subBlockNum; ++i)
{
sortLabelInSubBlock(subBlockArray[i]);
}
}
*/

// DELETE
// 取消：直接复制指针，不要重复分配空间
/*
// 释放子块的动态内存空间
void destroySubBlock(sSubBlock &subBlock)
{
delete[]subBlock.subBlockNodeArray;
}
*/
/********************************** END: class subBlock **********************************/


/********************************* BEGIN: class sWorkset sWorkseg *********************************/
// DELETE
// 无用的sWorkset成员函数
/*
void sWorkset::setBeg(int subBlockIdxX, int subBlockIdxY)
{
int beg = 0;
for (int j = 0; j <= subBlockIdxY; ++j)
{
for (int i = 0; i < subBlockIdxX; ++i)
{
beg += computeSubset(matrixRate, i, j, subBlockLen);
}
}
this->beg = beg;
}

void sWorkset::setEnd(int subBlockIdxX, int subBlockIdxY)
{
this->end = this->beg + computeSubset(matrixRate, subBlockIdxX, subBlockIdxY, subBlockLen);
}
*/

// DELETE
// 利用记录的数组matrixSubset得出单个workset的beg和end
/*
void setWorkset(int bid)
{
int subBlockIdxX;
int subBlockIdxY;
getBlockXY(bid, subBlockIdxX, subBlockIdxY);

int beg = 0;
for (int i = 0; i <= subBlockIdxX; ++i)
{
for (int j = 0; j < subBlockIdxY; ++j)
{
beg += matrixSubset[i][j];
}
}
worksetArray[bid].beg = beg;
worksetArray[bid].end = beg + matrixSubset[subBlockIdxX][subBlockIdxY];
}
*/

// NEW
// 扫描bid排序后的rateNodeArray, 得出每个workset的beg和end，保存到worksetArray
// 同时统计每个subset的值,保存到subsetArray
void setWorkset()
{
    if(NNZ <= 0)
    {
        return;
    }

    int beg = 0;
    int curBid = 0;

    for(int i = 0; i < NNZ; ++i)
    {
        curBid = rateNodeArray[i].bid;
        beg = i;

        while(i < NNZ && (curBid == rateNodeArray[i].bid))  // 至少执行一次
        {
            ++i;
        }

        worksetArray[curBid].beg = beg;
        worksetArray[curBid].end = i;
        subsetArray[curBid] = i - beg;
        --i; // for循环有++i
    }
}

// 扫描bid_label排序后的rateNodeArray, 得出每个workseg的from和to，保存到mWorkseg
void setWorkseg()
{
    if(NNZ <= 0)
    {
        return;
    }

    int beg = 0;
    int end = 0;
    int from = 0;
    int to = 0;
    int curLabel;

    for(int curBid = 0; curBid < subBlockNum; ++curBid)     // 每个子块curBid
    {
        beg = worksetArray[curBid].beg;
        end = worksetArray[curBid].end;

        for(int i = beg; i < end; ++i)                      // 子块内
        {
            curLabel = rateNodeArray[i].label;
            from = i;

            while(i < NNZ && (curLabel == rateNodeArray[i].label))  // 至少执行一次
            {
                ++i;
            }

            (mWorkseg + curBid * subBlockLen)[curLabel].from = from;    // .的优先级比*高，用*要加括号 (*(*(mWorkseg + curBid*subBlockLen + curLabel))).from
            (mWorkseg + curBid * subBlockLen)[curLabel].to = i;
            //mWorkseg[curBid][curLabel].from = from;
            //mWorkseg[curBid][curLabel].to = i;
            --i; // for循环有++i
        }
    }
}
/********************************** END: class sWorkset sWorkseg **********************************/


// 数组array内存分配
template <typename T>
void newArray(T* &array, int n, int val)
{
    array = new T[n];
    memset(array, val, n * sizeof(T));
}

// 矩阵m内存分配
template <typename T>
void newMatrix(T** &m, int rowNum, int colNum, int val)
{
    m = new T *[rowNum];

    for(int i = 0; i < rowNum; ++i)
    {
        m[i] = new T[colNum];
        memset(m[i], val, colNum * sizeof(T));
    }
}

// 1维的矩阵m内存分配, 并随机初始化
template <typename T>
void newMatrixRandom1D(T* &m, int rowNum, int colNum)
{
    m = new T[rowNum * colNum];

    for(int i = 0; i < rowNum; ++i)
    {
        for(int j = 0; j < colNum; ++j)
        {
            *(m + i * colNum + j) = (rand() % RAND_MAX) / (typeRate)(RAND_MAX);
        }
    }
}

// 数组array内存销毁
template <typename T>
void deleteArray(T* &array, int n, int val)
{
    delete[]array;
}

// 矩阵m内存销毁
template <typename T>
void deleteMatrix(T** &m, int rowNum, int colNum)
{
    for(int i = 0; i < rowNum; ++i)
    {
        delete[] m[i];
    }

    delete[] m;
}

// 随机初始化矩阵(0~1)
void randomInitMatrix(typeRate **m, int rowNum, int colNum)
{
    for(int i = 0; i < rowNum; ++i)
    {
        for(int j = 0; j < colNum; ++j)
        {
            m[i][j] = (rand() % RAND_MAX) / (typeRate)(RAND_MAX);
        }
    }
}


/********************************* BEGIN: shuffle matrix *********************************/
// debug:
void printNodeArrayAsMatrix()
{
    sortRateNodeArrayUserItem();
    int curUser;

    for(int i = 0; i < NNZ; ++i)
    {
        curUser = rateNodeArray[i].u;
        cout << curUser << ": ";

        while(i < NNZ && (curUser == rateNodeArray[i].u))   // 至少执行一次
        {
            cout << rateNodeArray[i].i << " ";
            ++i;
        }

        cout << endl;
        --i;
    }
}

// 输出子块nnz的max_diff
int getMaxDiff()
{
    int max = *max_element(subsetArray, subsetArray + subBlockNum);
    int min = *min_element(subsetArray, subsetArray + subBlockNum);
    //printList(subsetArray, subBlockNum);
    return max - min;
}

// 输出子块均匀度
double getEvenness()
{
    int max_diff = getMaxDiff();
    double avg = (double)NNZ / subBlockNum;
    return (double)avg / (avg + max_diff);
}

// 在乱序的rateNodeArray计算subsetArray，省下排序时间
void computeSubsetArray()
{
    int curBid;

    for(int i = 0; i < NNZ; ++i)
    {
        setSubBlockIdx(rateNodeArray[i]);
        curBid = rateNodeArray[i].bid;
        ++subsetArray[curBid];
    }
}

// for random_shuffle
// random generator function:
ptrdiff_t myrandom(ptrdiff_t i)
{
    return rand() % i;
}
// pointer object to it:
ptrdiff_t(*p_myrandom)(ptrdiff_t) = myrandom;


// 矩阵行shuffle NNZ版本(基于任意顺序的rateNodeArray)
// note: 要保存bestPermRow，用于复原, permRow大小为M, 初始化为{1, 2, ..., M}
void rowShuffle(vector<int> &curPermRow, vector<int> &bestPermRow)
{
    srand((unsigned)time(NULL));
    //cout << "rowShuffle called: " << endl;
    //int userIdx;
    /*
    for (int i = 0; i < M; ++i)
    {
    permRow[i] = i;
    }
    */
	random_shuffle(curPermRow.begin(), curPermRow.end(), p_myrandom);
	//#pragma omp parallel for
	#pragma omp parallel num_threads(ompNumThread)
	{
		#pragma omp for
		for (int i = 0; i < NNZ; ++i)
		{
			//question: 竞争?
			/*
			userIdx = rateNodeArray[i].u - 1;
			rateNodeArray[i].u = permRow[userIdx];  // FIXME: 假设数据集idx： 1~M
			*/
			rateNodeArray[i].u = curPermRow[rateNodeArray[i].u - 1];    // FIXME: 假设数据集idx： 1~M
		}
	}

    //#pragma omp parallel for
	#pragma omp parallel num_threads(ompNumThread)
	{
		#pragma omp for
		for (int i = 0; i < M; ++i)
		{
			bestPermRow[i] = curPermRow[bestPermRow[i] - 1];
		}
	}
}

// 矩阵列shuffle NNZ版本(基于任意顺序的rateNodeArray)
// note: 要保存bestPermColumn，用于复原, permColumn大小为N, 初始化为{1, 2, ..., N}
void columnShuffle(vector<int> &curPermColumn, vector<int> &bestPermColumn)
{
    srand((unsigned)time(NULL));
    //cout << "columnShuffle called: " << endl;
    //int itemIdx;
    /*
    for (int i = 0; i < N; ++i)
    {
    permColumn[i] = i;
    }
    */
	random_shuffle(curPermColumn.begin(), curPermColumn.end(), p_myrandom);
	// debug:
	//printList(&permColumn[0], N);
	//#pragma omp parallel for
	#pragma omp parallel num_threads(ompNumThread)
	{
		#pragma omp for
		for (int i = 0; i < NNZ; ++i)
		{
			//question: 竞争?
			/*
			itemIdx = rateNodeArray[i].i - 1;
			rateNodeArray[i].i = permColumn[itemIdx];   // FIXME: 假设数据集idx： 1~N
			*/
			rateNodeArray[i].i = curPermColumn[rateNodeArray[i].i - 1]; // FIXME: 假设数据集idx： 1~N
		}
	}

    //#pragma omp parallel for
	#pragma omp parallel num_threads(ompNumThread)
	{
		#pragma omp for
		for (int i = 0; i < N; ++i)
		{
			bestPermColumn[i] = curPermColumn[bestPermColumn[i] - 1];
		}
	}

    // debug:
    //printNodeArrayAsMatrix();
}

// 矩阵随机变换(最优解)
void matrixShuffle()
{
    rateNodeVector_backup = rateNodeVector;
    // debug:
    //printVar("NNZ", NNZ);
    //sortRateNodeArrayBid();
    //memset(worksetArray, 0, subBlockNum*sizeof(sWorkset));
    //memset(subsetArray, 0, subBlockNum*sizeof(int));
    //setWorkset();
    // debug:
    /*
    cout << "subsetArray: ";
    printList(subsetArray, subBlockNum);
    */
    vector<int> curPermRow;     // 当前所应用的变换策略
    vector<int> curPermColumn;
    vector<int> bestPermRow;    // 模拟{1, 2, 3,...,M}开始变换，可以从原始"一步"变成rateNodeArray当前状态的序列
    vector<int> bestPermColumn;
    // 初始化permRow, permColumn
    int i;

    for(i = 0; (i < M) && (i < N); ++i)
    {
        curPermRow.push_back(i + 1);
        curPermColumn.push_back(i + 1);
        bestPermRow.push_back(i + 1);
        bestPermColumn.push_back(i + 1);
    }

    while(i < M)
    {
        curPermRow.push_back(i + 1);
        bestPermRow.push_back(i + 1);
        ++i;
    }

    while(i < N)
    {
        curPermColumn.push_back(i + 1);
        bestPermColumn.push_back(i + 1);
        ++i;
    }

    int min_diff = NNZ;
    int cur_diff;
    double avg = ((double)NNZ / subBlockNum);
    printVar("avg", avg);
    auto start = system_clock::now();

    for(int i = 0; i < MAX_SHUFFLE; ++i)
    {
        // 交给computeSubsetArray()避免多次遍历
        //resetAllNode_blockIdx();  // 变换后重置bid
        //sortRateNodeArrayBid();
        //memset(worksetArray, 0, subBlockNum*sizeof(sWorkset));
        memset(subsetArray, 0, subBlockNum * sizeof(int));
        //setWorkset();
        computeSubsetArray();
        // debug:
        /*
        cout << "subsetArray: ";
        printList(subsetArray, subBlockNum);
        */
        cur_diff = getMaxDiff();

        if(cur_diff < min_diff)
        {
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            cout << "curTime: \t" << double(duration.count()) * microseconds::period::num / microseconds::period::den << " seconds" << endl;
            min_diff = cur_diff;
            permRow.assign(bestPermRow.begin(), bestPermRow.end());
            permColumn.assign(bestPermColumn.begin(), bestPermColumn.end());
            //printVar("cur_diff", cur_diff);
            double ED = avg / (avg + cur_diff);
            printVar("cur_diff", cur_diff);
            printVar("ED", ED);
            // debug:
            /*
            cout << "subsetArray: ";
            printList(subsetArray, subBlockNum);
            cout << "bestPermRow: ";
            printList(&bestPermRow[0], M);
            cout << "bestPermColumn: ";
            printList(&bestPermColumn[0], N);
            */
        }
        // 保存之前的best再变换，最后一次变换不采用
        rowShuffle(curPermRow, bestPermRow);
        columnShuffle(curPermColumn, bestPermColumn);
    }

    matrixShuffleApply();
    resetAllNode_blockIdx_label();
}

// rateNodeArray指向备份矩阵，应用最终的变换
void matrixShuffleApply()
{
    rateNodeArray = &rateNodeVector_backup[0];

    if(permRow.empty() || permColumn.empty())
    {
        return;
    }

    //应用最终的变换
    //#pragma omp parallel for
	#pragma omp parallel num_threads(ompNumThread)
	{
		#pragma omp for
		for (int i = 0; i < NNZ; ++i)
		{
			rateNodeArray[i].u = permRow[rateNodeArray[i].u - 1];   // FIXME: 假设数据集idx： 1~M
			rateNodeArray[i].i = permColumn[rateNodeArray[i].i - 1];    // FIXME: 假设数据集idx： 1~N
		}
	}

#pragma omp parallel num_threads(ompNumThread)
	{
#pragma omp for
		for (int i = 0; i < NNZ_test; ++i)
		{
			rateNodeArrayTest[i].u = permRow[rateNodeArrayTest[i].u - 1];   // FIXME: 假设数据集idx： 1~M
			rateNodeArrayTest[i].i = permColumn[rateNodeArrayTest[i].i - 1];    // FIXME: 假设数据集idx： 1~N
		}
	}
}

// 暂时不用
/*
// TO-DO
// 矩阵变换后，还原matrixUser, matrixItem到正确的行、列，用于预测。
void matrixShuffleRecover()
{
vector<int> hashVectorRow(M+1);
vector<int> hashVectorCol(N+1);
for (int i = 1; i <= M; ++i)
{
hashVectorRow[permRow[i-1]] = i;
}
for (int i = 1; i <= N; ++i)
{
hashVectorCol[permColumn[i-1]] = i;
}

//TO-DO
}
*/







// DELETE: 不用排序直接变换也可以
// 矩阵行shuffle NNZ版本(基于rateNodeArray（原始输入，排序前），即按userIdx排序)
/*
void rowShuffle()
{
cout << "rowShuffle called: " << endl;

vector<int> permVec(M);
for (int i = 0; i < M; ++i)
{
permVec[i] = i;
}
random_shuffle(permVec.begin(), permVec.end());

for (int i = 0; i < NNZ; ++i)
{
int userIdx = rateNodeArray[i].u;
while ((i < NNZ) && (userIdx == rateNodeArray[i].u))
{
rateNodeArray[i].u = permVec[i];
++i;
}
--i;    // for会++i
}
}
*/

// DELETE: 矩阵shuffle原始版 (随机生成矩阵的函数可以用下)
#if 0
// 矩阵行shuffle
void rowShuffle(typeRate **a, int rowNum, int colNum)
{
    cout << "rowShuffle called: " << endl;
    int *newPos = new int[rowNum];
    int *permList = new int[rowNum];

    //int testList[5] = { 0, 1, 2, 3, 4 };
    for(int i = 0; i < rowNum; ++i)
    {
        newPos[i] = i;
        permList[i] = i;
    }

    random_shuffle(permList, permList + rowNum);

    //cout << "rand_perm list: ";
    //printList(permList, rowNum);
    //cout << endl;
    for(int i = 0; i < rowNum - 1; ++i)
    {
        int newPosition = newPos[permList[i]];

        for(int j = 0; j < colNum; ++j)
        {
            //swap(testList[i], testList[newPosition]);
            swap(a[i][j], a[newPosition][j]);
        }

        newPos[i] = permList[i];
    }

    //printList(testList, rowNum);
    cout << endl;
    delete newPos;
    delete permList;
}

// 矩阵的行/列 shuffle (调用rowShuffle)
void randomShuffleMatrix(typeRate **m, int rowNum, int columnNum)
{
    cout << "randomShuffleMatrix called: " << endl;
    // 列变换
    transposeMatrix(m, rowNum, columnNum);
    //printMatrix(m, rowNum, columnNum);
    rowShuffle(m, rowNum, columnNum);
    //printMatrix(m, rowNum, columnNum);
    // 复原
    transposeMatrix(m, rowNum, columnNum);
    // 行变换
    //printMatrix(m, rowNum, columnNum);
    rowShuffle(m, rowNum, columnNum);
    //printMatrix(m, rowNum, columnNum);
    cout << endl;
}

// 矩阵分块，并统计每块包含的元素个数
int blockMatrix(typeRate **a, int rowNums, int blockLen)
{
    cout << "blockMatrix called:" << endl;
    const int blockSize = blockLen; // element_num_per_block = blockSize * blockSize;
    const int divider = rowNums / blockSize;
    int *b = new int[divider * divider];
    memset(b, 0, divider * divider * sizeof(int));

    // block
    for(int i = 0; i < rowNums; ++i)
    {
        for(int j = 0; j < rowNums; ++j)
        {
            //cout << a[i][j] << " ";
            if(a[i][j] != 0)
            {
                int bi = i / blockSize;
                int bj = j / blockSize;
                ++b[bi * divider + bj];
            }
        }

        //cout << endl;
    }

    // count
    int max = INT_MIN;
    int min = INT_MAX;
    int avg = 0;

    //cout << endl;
    for(int i = 0; i < divider; ++i)
    {
        for(int j = 0; j < divider; ++j)
        {
            cout << b[i * divider + j] << " ";
            max = (max > b[i * divider + j]) ? max : b[i * divider + j];
            min = (min < b[i * divider + j]) ? min : b[i * divider + j];
            avg += b[i * divider + j];
        }

        cout << endl;
    }

    cout << "max: " << max << endl;
    cout << "min: " << min << endl;
    cout << "avg: " << (avg / (divider * divider)) << endl;
    cout << "diff: " << (max - min) << endl;
    delete[]b;
    return (max - min);
}

// testShuffle
void randomGenerateMatrix()
{
    srand((unsigned)time(NULL));
    const int MAX = 1000;   // element_num_all = MAX * MAX;
    typeRate **a = new typeRate*[MAX];
    int nnz = 0;

    for(int i = 0; i < MAX; ++i)
    {
        a[i] = new typeRate[MAX];

        for(int j = 0; j < MAX; ++j)
        {
            if(rand() % 100 == 0)
            {
                a[i][j] = (rand() % 5) + 1; // rating: 1 ~ 5
                ++nnz;
            }
            else
            {
                a[i][j] = 0;
            }
        }
    }

    cout << "before shuffle, block result: " << endl;
    blockMatrix(a, MAX, 50);
    int min_diff = INT_MAX;

    for(int i = 0; i < 10; ++i)
    {
        randomShuffleMatrix(a, MAX, MAX);
        cout << "after shuffle, block result: " << endl;
        int diff = blockMatrix(a, MAX, 50);
        min_diff = (min_diff < diff) ? min_diff : diff;
    }

    cout << "min_diff: " << min_diff << endl;

    /*
    int max = INT_MIN;
    int min = INT_MAX;
    for (int i = 0; i < 100; ++i)
    {
    randomShuffleMatrix(a, MAX);

    int diff = blockMatrix(a, MAX);
    max = (diff > max) ? diff : max;
    min = (diff < max) ? diff : max;

    }
    */

    //cout << max << "\t" << min << endl;

    for(int i = 0; i < MAX; ++i)
    {
        delete[] a[i];
    }

    delete[]a;
}
#endif
/********************************** END: shuffle matrix **********************************/

// DELETE
// 检查子块bid是否越界
/*
int checkSubBlockBoundary(int bid)
{
int subBlockIdxX;
int subBlockIdxY;
getBlockXY(bid, subBlockIdxX, subBlockIdxY);

if ((subBlockIdxX * subBlockLen >= M) || (subBlockIdxY * subBlockLen >= N))
{
return 1;
}
else
{
return 0;
}
}
*/

// 计算子块x, y对应的bid，其中子块大小为: subBlockLen * subBlockLen
int computeSubBlockID(int subBlockNumL, int subBlockIdxX, int subBlockIdxY)
{
    return subBlockIdxX * subBlockNumL + subBlockIdxY;
}

// 计算bid对应的子块x,y下标 (0开始)
void getBlockXY(int bid, int &x, int &y)
{
    x = (bid / subBlockNumL);
    y = bid % subBlockNumL;
}

// 计算Rui所属的子块x,y下标
void getBlockXY(int u, int i, int &x, int &y)
{
    x = u / subBlockLen;
    y = i / subBlockLen;
}


// 记录子块b_xy的ID: bid到二维数组pattern(s, t)
// 子块b_xy 是第s种模式中的第t个子块, 则把computeSubBlockID(z, x, y)的结果放入pattern(s,t)
void setPattern(int s, int t, int x, int y)
{
    *(matrixPattern + s * subBlockNumL + t) = computeSubBlockID(subBlockNumL, x, y);
}

// 记录子块b_xy的ID: bid到二维数组pattern(s, t)
// 子块bid是第s种模式中的第t个子块, 则把bid放入pattern(s,t)
void setPattern(int s, int t, int bid)
{
    *(matrixPattern + s * subBlockNumL + t) = bid;
}

// DELETE
// 原subset矩阵的计算
#if 0
// 返回：subset(x, y)
// 子块 b_xy 包含的评价值个数(非零元素)
int computeSubset(int subBlockIdxX, int subBlockIdxY)
{
    int bid = computeSubBlockID(subBlockLen, subBlockIdxX, subBlockIdxY);
    int checkResult = checkSubBlockBoundary(bid);

    // 完全越界
    if(checkResult == 1)
    {
        return 0;
    }

    int cnt = 0;
    int leftUpIdxRow = subBlockIdxX * subBlockLen;
    int leftUpIdxCol = subBlockIdxY * subBlockLen;
    int rightDownIdxRow = leftUpIdxRow + subBlockLen - 1;
    int rightDownIdxCol = leftUpIdxCol + subBlockLen - 1;

    for(int i = leftUpIdxRow; i <= min(rightDownIdxRow, M - 1); ++i)
    {
        for(int j = leftUpIdxCol; j <= min(rightDownIdxCol, N - 1); ++j)
        {
            if(matrixRate[i][j] != 0)  // fabs(m[i][j]) > 1e-8
            {
                ++cnt;
            }
        }
    }

    return cnt;
}

// 记录所有subset(x, y)到全局二维数组matrixSubset
// 子块 b_xy 包含的评价值个数(非零元素)
void computeAllSubset()
{
    int cnt = 0;

    for(int subBlockIdxX = 0; subBlockIdxX < subBlockNumL; ++subBlockIdxX)
    {
        for(int subBlockIdxY = 0; subBlockIdxY < subBlockNumL; ++subBlockIdxY)
        {
            /*
            cnt += computeSubset(matrixRate, subBlockIdxX, subBlockIdxY, subBlockLen);
            matrixSubset[subBlockIdxX][subBlockIdxY] = cnt;
            */
            matrixSubset[subBlockIdxX][subBlockIdxY] = computeSubset(subBlockIdxX, subBlockIdxY);
        }
    }
}
#endif


// 对子块b_xy所有元素(非0)进行label
// DELETE:
/*
void labelSubBlock(int subBlockIdxX, int subBlockIdxY)
{
int leftUpIdxRow = subBlockIdxX * subBlockLen;
int leftUpIdxCol = subBlockIdxY * subBlockLen;
int rightDownIdxRow = leftUpIdxRow + subBlockLen - 1;
int rightDownIdxCol = leftUpIdxCol + subBlockLen - 1;
for (int i = 0; leftUpIdxRow + i <= rightDownIdxRow; ++i)
{
for (int j = 0; leftUpIdxCol + j <= rightDownIdxCol; ++j)
{
if (matrixRate[leftUpIdxRow + i][leftUpIdxCol + j] != 0) // fabs(matrixRate[leftUpIdxRow + i][leftUpIdxCol + j]) > 1e-8
{
int label = (subBlockLen - i + j) % subBlockLen;

}
}
}
}
*/

// debug:
// 输出user item
void printNodeArrayMatrix()
{
    sortRateNodeArrayUserItem();
    map<int, vector<int>> imap;
    int userIdx;
    int itemIdx;

    for(int idx = 0; idx < NNZ; ++idx)
    {
        userIdx = rateNodeArray[idx].u;
        itemIdx = rateNodeArray[idx].i;
        imap[userIdx].push_back(itemIdx);
    }

    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < imap[i].size(); ++j)
        {
            cout << imap[i][j] << "\t";
        }

        cout << endl;
    }
}






// debug:
void unitTest()
{
    CALL_FUN_TIME(initParameter())
    CALL_FUN_TIME(readFile(inputFile))

    if(testReadFile == 1) return;

    /*
    sortRateNodeArrayBid();
    for (int i = 0; i < rateNodeVector.size(); ++i)
    {
    printRateNode(rateNodeArray[i]);
    printLine();
    }
    */
    //initAllData();
    CALL_FUN_TIME(initAllData())

    //matrixShuffle();
    if(isShuffle != 0)
    {
        CALL_FUN_TIME(matrixShuffle())
    }
	if (testShuffle == 1) return;

    //sortRateNodeArrayBid();
    CALL_FUN_TIME(sortRateNodeArrayBid())
    // 矩阵变换后需重置
    memset(worksetArray, 0, subBlockNum * sizeof(sWorkset));
    memset(subsetArray, 0, subBlockNum * sizeof(int));
    CALL_FUN_TIME(setWorkset())     //setWorkset();
    CALL_FUN_TIME(setWorkseg())     //setWorkseg();
    CALL_FUN_TIME(setAllPattern())  //setAllPattern();
    fclose(stdin);
    freopen("CON", "r", stdin);   //"CON"代表控制台
    printLine();
    // debug:
    /*
    printMatrixPattern();
    printLine();
    printWorksetArray();
    printLine();
    printMatrixWorkseg();
    printLine();

    for (int i = 0; i < rateNodeVector.size(); ++i)
    {
        cout << "i = " << i << endl;
        printRateNode(rateNodeArray[i]);
        printLine();
    }
    */
    //printList(&permRow[0], M);
    //printList(&permColumn[0], N);
    //printNodeArrayAsMatrix();
    ///*
    CALL_FUN_TIME(solveByGPU(rateNodeArray, matrixUser, matrixItem, worksetArray,
                             mWorkseg, matrixPattern, subBlockNumL, subBlockLen,
                             lambda, gamma,
                             NNZ))
    //*/
    //callKernel(rateNodeArray, matrixUser, matrixItem, M, N, K, worksetArray, mWorkseg, matrixPattern, subBlockNumL, subBlockLen, NNZ);
    //debug:
    /*
    typeRate *matrixPredict = new typeRate[M*N];
    memset(matrixPredict, 0, M*N*sizeof(typeRate));
    matrixMultiply(matrixUser, matrixItem, matrixPredict);

    printMatrix(matrixUser, M, K);
    printMatrix(matrixItem, N, K);
    printMatrix(matrixPredict, M, N);
    printNodeArrayMatrix();
    */
}


// 总执行函数
void execute()
{
    // 初试化、输入、内存分配(array matrix并及时memset为0)
    initParameter();
    readFile(inputFile); // 评分矩阵
    initAllData();
    // CPU处理
    matrixShuffle();
    sortRateNodeArrayBid();
    // 矩阵变换后需重置
    memset(worksetArray, 0, subBlockNum * sizeof(sWorkset));
    memset(subsetArray, 0, subBlockNum * sizeof(int));
    setWorkset();
    setWorkseg();
    setAllPattern();
    // 调用GPU程序
    // 矩阵重置: 根据permRow, permCol还原matrixUser, matrixItem
    // 保存结果 (model)
    // 如果执行方式为：预测
    // 预测并保存结果
    // 内存free
}

// sgd CPU update
void sgd_CPU()
{
    for(int iter = 0; iter < MAX_ITER; ++iter)
    {
        for(int i = 0; i < NNZ; ++i)
        {
            int userIdx = rateNodeArray[i].u - 1;
            int itemIdx = rateNodeArray[i].i - 1;
            typeRate rate = rateNodeVector[i].rate;
            typeRate predict = innerProduct(matrixUser, matrixItem, userIdx, itemIdx);
            typeRate err = rate - predict;

            for(int k = 0; k < K; ++k)
            {
                typeRate tmp = (*(matrixUser + userIdx * K + k));
                (*(matrixUser + userIdx * K + k)) += (gamma * (err * (*(matrixItem + itemIdx * K + k)) - lambda * tmp));
                (*(matrixItem + itemIdx * K + k)) += (gamma * (err * tmp - lambda * (*(matrixItem + itemIdx * K + k))));
            }
        }

        double err_sum = 0;

        for(int i = 0; i < NNZ; ++i)
        {
            int userIdx = rateNodeArray[i].u - 1;
            int itemIdx = rateNodeArray[i].i - 1;
            typeRate rate = rateNodeArray[i].rate;
            typeRate predict = innerProduct(matrixUser, matrixItem, userIdx, itemIdx);
            typeRate err = rate - predict;
            err_sum += pow(err, 2);
        }

        typeRate rmse = sqrt(err_sum / NNZ);
        cout << "iter: " << iter << "\t rmse: " << rmse << endl;
    }
}

// FGMF CPU版本
void FGMF_CPU()
{
    CALL_FUN_TIME(initParameter())
    CALL_FUN_TIME(readFile(inputFile))
    /*
    sortRateNodeArrayBid();
    for (int i = 0; i < rateNodeVector.size(); ++i)
    {
    printRateNode(rateNodeArray[i]);
    printLine();
    }
    */
    //initAllData();
    CALL_FUN_TIME(initAllData())
    //matrixShuffle();
    CALL_FUN_TIME(matrixShuffle())
    //sortRateNodeArrayBid();
    CALL_FUN_TIME(sortRateNodeArrayBid())
    // 矩阵变换后需重置
    memset(worksetArray, 0, subBlockNum * sizeof(sWorkset));
    memset(subsetArray, 0, subBlockNum * sizeof(int));
    CALL_FUN_TIME(setWorkset())     //setWorkset();
    CALL_FUN_TIME(setWorkseg())     //setWorkseg();
    CALL_FUN_TIME(setAllPattern())  //setAllPattern();
    CALL_FUN_TIME(sgd_CPU())
}