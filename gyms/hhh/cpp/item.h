#ifndef ITEM_H
#define ITEM_H

namespace hh
{

typedef unsigned long item_id;
typedef unsigned int counter;

struct entry
{
	item_id id;
	counter count;
	counter error;

	inline entry(const item_id& id, const counter& count, const counter& error = 0)
	: id(id), count(count), error(error)
	{}
};

}

#endif
