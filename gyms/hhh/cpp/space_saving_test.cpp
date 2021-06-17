#include <cstdlib>
#include <iostream>

#include "measure.h"
#include "space_saving.h"

using namespace std;
using namespace hh;

void print_result(const space_saving& spcs, const counter& threshold)
{
	for (auto& hh : spcs.query(threshold))
		cout << hh->id << " " << hh->count << " " << hh->error << ", ";

	cout << "\b\b " << endl;
}

int main(void)
{
	space_saving spcs(0.2);

	spcs.update(0x1, 1);
	spcs.update(0x2, 5);
	spcs.update(0x3, 9);
	spcs.update(0x4, 1);
	spcs.update(0x5, 1);
	print_result(spcs, 0);
	spcs.update(0x6, 1);
	print_result(spcs, 0);
	spcs.update(0x4, 1);
	print_result(spcs, 0);
	spcs.update(0x5, 1);
	print_result(spcs, 0);
	spcs.update(0x3, 9);
	print_result(spcs, 0);
	spcs.update(0x1, 1);
	print_result(spcs, 0);
	spcs.update(0x5, 1);
	print_result(spcs, 0);
	spcs.update(0x6, 2);
	print_result(spcs, 0);
	print_result(spcs, 4);

	const static int paralellism = 8;

	shared_ptr<space_saving> spcsm[paralellism];
	counter N[paralellism];

	start_measure();

	for (int i = 0; i < paralellism; ++i)
	{
		spcsm[i] = shared_ptr<space_saving>(new space_saving(0.001));
		N[i] = 0;
	}

	int i, j, k;

	#pragma omp parallel for private(i)
	for (i = 0; i < paralellism; ++i)
	for (j = 0; j < 1000; ++j)
	{
		for (k = 0; k < 10000; ++k)
		{
			counter r = rand() % 1000;
			N[i] += r;
			spcsm[i]->update(rand() % 1000, r);
		}

		spcsm[i]->query(N[i] * 0.2);
	}

	stop_measure("");
}
