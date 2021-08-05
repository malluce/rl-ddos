#ifndef ITEM_H
#define ITEM_H

namespace hh
{

struct item
{
	typedef unsigned long id;
	typedef unsigned int counter;

	inline item()
	: item(0, 0, 0)
	{}

	// items track one-sided errors (overestimation)
	inline item(const id& item_id, const counter& count, const counter& error = 0)
	: item_id(item_id), count(count), error(error)
	{}

	id item_id;
	counter count;
	counter error;
};

}

#endif
