#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <string>
#include <vector>

#include "fgmf.h"
#include "basic_func.h"

#include "sgd.h"

using namespace std;
extern string inputFile;
string outputFile;

extern int experimentVar1;

// test label
void label_matrix(int *matrixA, int N)
{
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            *(matrixA + i * N + j) = (N - i + j) % N;
        }
    }
}

/*
void callGPU()
{
int size = 20;
int *a = new int[size];
int *b = new int[size];
int *c = new int[size];

for (int i = 0; i < size; ++i)
{
a[i] = 1;
b[i] = 2;
}

//solveByGPU(a, b, c, size);


printList(c, size);
delete[]a;
delete[]b;
delete[]c;
getchar();
}
*/


void idMap(string fileName)
{
	// serial id Map
	ifstream inputFile(fileName);
	int userIdx, mapUserIdx;
	int itemIdx, mapItemIdx;
	double rate;
	unordered_map<int, int> userMap;
	unordered_map<int, int> itemMap;
	long userCnt = 0;
	long itemCnt = 0;
	while (!inputFile.eof())
	{
		inputFile >> userIdx >> itemIdx >> rate;
		// bug: empty line keeps userIdx/itemIdx the last value, which cause replica of last line
		if (userMap.find(userIdx) == userMap.end())
		{
			mapUserIdx = userCnt;
			userMap[userIdx] = (userCnt++);
		}
		else
		{
			mapUserIdx = userMap[userIdx];
		}
		if (itemMap.find(itemIdx) == itemMap.end())
		{
			mapItemIdx = itemCnt;
			itemMap[itemIdx] = (itemCnt++);
		}
		else
		{
			mapItemIdx = itemMap[itemIdx];
		}
		cout << mapUserIdx + 1 << " " << mapItemIdx + 1 << " " << rate << endl;
	}
}


void test()
{
    //string outputFile = "output/console_output.txt";
	if (outputFile.empty()) {
		outputFile = "output/result_" + inputFile.substr(inputFile.find('/') + 1);
		stringstream ss;
		ss << (int)time(NULL);
		outputFile += ss.str();
	}
    freopen(outputFile.c_str(), "w", stdout);
	//idMap(inputFile);
    unitTest();
}


int main(int argc, char** argv)
{
    srand((unsigned)time(NULL));

    if(argc > 1)
    {
        inputFile = argv[1];
    }

    if(argc > 2)
    {
        outputFile = argv[2];
    }

	if(argc > 3)
	{
		experimentVar1 = argv[2];
	}

#if 0
    string inputFile = "input.txt";
    string outputFile = "output.txt";
    freopen(inputFile.c_str(), "r", stdin);
    freopen(outputFile.c_str(), "w", stdout);
#endif
    //callGPU();
    test();
    //execute();
    //getchar();
    fclose(stdout);
    //fclose(stdin);
    return 0;
}

