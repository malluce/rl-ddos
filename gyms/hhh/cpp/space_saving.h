#ifndef SPACE_SAVING_H
#define SPACE_SAVING_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "item.h"

namespace hh
{

class space_saving
{
private:
	typedef std::shared_ptr<entry> entry_ptr;
	typedef std::unordered_map<item_id, entry_ptr> item_map;
	typedef std::shared_ptr<item_map> item_map_ptr;
	typedef item_map bucket_map;
	typedef std::shared_ptr<bucket_map> bucket_map_ptr;
	typedef std::map<counter, bucket_map_ptr> counter_map;
	typedef std::shared_ptr<counter_map> counter_map_ptr;

public:
	typedef counter_map::const_iterator result_iterator;
	typedef std::pair<result_iterator, result_iterator> result_type;

	space_saving(double epsilon);

	void update(const item_id& id, const counter& count = 1);
	result_type query(const counter& threshold) const;

private:
	bucket_map_ptr get_bucket(const counter& count);
	void remove_entry(const counter& count, const item_id& id);

	double epsilon;
	counter_map::size_type max_items;
	counter_map_ptr counters;
	item_map_ptr items;
};

}

#endif
