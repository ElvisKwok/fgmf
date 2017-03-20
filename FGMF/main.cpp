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

void test()
{
    //string outputFile = "output/console_output.txt";
    outputFile = "output/result_" + inputFile.substr(inputFile.find('/') + 1);
    stringstream ss;
    ss << (int)time(NULL);
    outputFile += ss.str();
    //freopen(outputFile.c_str(), "w", stdout);
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

