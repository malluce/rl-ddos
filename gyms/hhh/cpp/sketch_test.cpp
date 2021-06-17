#include <iostream>

#include "measure.h"
#include "sketch_hhh.h"

#include <iomanip>

using namespace std;
using namespace hhh;

int main(void)
{
	sketch_hhh sk(0.001);

	sk.update(0x0,  5);
	sk.update(0x1, 10);
	sk.update(0x2, 10);
	sk.update(0x2, 10);
	sk.update(0x4,  5);
	sk.update(0x7,  5);
	sk.update(0xffffffff, 5);

	for (auto& h : sk.query(0.2))
		cout << h.id << ", " << h.len << ", "
			 << h.hi << ", " << h.lo
			 << endl;

	sketch_hhh sm(0.001);

	start_measure();

	for (int i = 0; i < 100; ++i)
	{
		for (int j = 0; j < 10000; ++j)
			sm.update(rand() % 1000, rand() % 1000);

		sm.query(0.1);
	}

	stop_measure("");
}
