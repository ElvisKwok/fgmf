#ifndef __PARALLEL_READ_FILE_H_
#define __PARALLEL_READ_FILE_H_

#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <boost/atomic.hpp>
#include <boost/thread/thread.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <windows.h>
#include <omp.h>
#include <deque>
#include <cstdlib>

#include "parameter.h"

using namespace::std;

typedef uint32_t vint;
typedef double dweight;
//typedef std::tuple<vint, vint, float> edge;

extern struct sRateNode;

int parallelReadFile(std::string &graphpath, std::vector<sRateNode> &edges);

#endif