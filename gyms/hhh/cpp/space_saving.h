#ifndef SPACE_SAVING_H
#define SPACE_SAVING_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <future>

#include "item.h"

namespace hh
{

using namespace ::boost;
using namespace ::boost::multi_index;

struct ids{};
struct counts{};

class space_saving
{
private:
	typedef multi_index_container<
		item,
		indexed_by<
			hashed_unique<tag<ids>, member<item, item::id, &item::item_id>>,
			ordered_non_unique<tag<counts>, member<item, item::counter, &item::count>>
		> 
	> ordered_item_map;

public:
	typedef ordered_item_map::index<counts>::type::iterator result_iterator;
	typedef std::pair<result_iterator, result_iterator> result_type;

	space_saving(double epsilon);

	void update(const item::id& item_id, const item::counter& count = 1);
	result_type query(const item::counter& threshold) const;
	void clear();

private:
	ordered_item_map::size_type max_items;
	ordered_item_map items;
};

}

#endif
