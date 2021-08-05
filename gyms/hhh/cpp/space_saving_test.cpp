#include <array>
#include <iostream>

#include "space_saving.h"
#include "measure.h"

using namespace std;
using namespace hh;

typedef item::counter counter;

int main(void)
{
	space_saving spcs(0.1);
	array<item::id, 39> ids = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		   1, 2, 3, 4, 5, 6, 7, 8, 9, // duplicates
		      2, 3, 4, 5, 6, 7, 8, 9, // duplicates
		         3, 4, 5, 6, 7, 8, 9, // duplicates
		11, 12, 11, 12, 13 // replacements
	};

	for (auto it = ids.begin(); it != ids.end(); ++it)
		spcs.update(*it);

	{
		auto [begin, end] = spcs.query(4);

		for (auto& it = begin; it != end; ++it)
			cout << it->item_id << ", " << it->count
				<< ", " << it->error << endl;
	}

	const static int paralellism = 4;

	space_saving* spcsm[paralellism];

	for (int i = 0; i < paralellism; ++i)
		spcsm[i] = new space_saving(0.01);

	int i;
	auto sum = counter(0);

	srand(42);
	start_measure();

	#pragma omp parallel for private(i, j, k)
	for (i = 0; i < paralellism; ++i)
	for (int j = 0; j < 10000; ++j)
	{
		auto N = counter(0);
		auto s = counter(0);

		for (int k = 0; k < 1000; ++k)
		{
			counter r = rand() % 1000;
			spcsm[i]->update(rand() % 1000, r);
			N += r;
		}

		auto [begin, end] = spcsm[i]->query(N * 0.011);

		// trigger iteration
		for (auto& it = begin; it != end; ++it)
			s += it->count;

		sum += s;
	}

	stop_measure("");
	cout << "sum " << sum << endl;

	for (int i = 0; i < paralellism; ++i)
		delete spcsm[i];

	return 0;
}
