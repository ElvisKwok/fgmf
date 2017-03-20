#ifndef __BASIC_FUNC_H_
#define ___BASIC_FUNC_H_


#include <iostream>
#include <vector>
#include <string>
#include "parameter.h"
using namespace std;

// read matrix from input file(or stdin)
void readMatrix(typeRate **m);

void printList(int *a, int len);

void printMatrix(typeRate *m, int rowNum, int colNum);

void transposeMatrix(typeRate *m, int row, int column);

void matrixMultiply(typeRate *matrixA, typeRate *matrixB, typeRate *matrixResult);

typeRate innerProduct(typeRate *matrixUser, typeRate *matrixItem, int userIdx, int itemIdx);
#endif