#ifndef MEASURE_H
#define MEASURE_H

#include <chrono>
#include <iostream>

using namespace std;

static chrono::time_point<std::chrono::steady_clock> start;

static inline void start_measure()
{	start = chrono::steady_clock::now(); }

static inline void stop_measure(const string& s)
{
	cout << s << ": "
		 << chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count()
		 << " µs" << endl;
}

#endif
