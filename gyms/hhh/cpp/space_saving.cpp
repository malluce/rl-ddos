#include "space_saving.h"

using namespace std;
using namespace hh;

space_saving::space_saving(double epsilon)
: epsilon(epsilon), max_items(counter_map::size_type(1.0 / epsilon)),
  counters(counter_map_ptr(new counter_map())),
  items(item_map_ptr(new item_map()))
{}

void space_saving::update(const item_id& id, const counter& count)
{
	auto item = items->find(id);

	if (item != items->cend())
	{
		/* item is tracked
		 * update item count and relink
		 */
		auto& e = item->second;

		remove_entry(e->count, id);

		e->count += count;

		auto b = get_bucket(e->count);

		b->insert({id, e});

		return;
	}

	/* item is not tracked
	 * replace item with lowest count and track error
	 */
	counter error = 0;

	if (items->size() == max_items)
	{
		auto  min_counter = counters->cbegin();
		auto& min_counter_bucket = min_counter->second;
		auto  first_entry = min_counter_bucket->cbegin();
		auto& id = first_entry->second->id;

		error = min_counter->first;
		items->erase(id);
		remove_entry(error, id);
	}

	// insert new item

	auto e = entry_ptr(new entry(id, count + error, error));
	auto b = get_bucket(count + error);

	items->insert({id, e});
	b->insert({id, e});
}

space_saving::result_type space_saving::query(const counter& threshold) const
{
	return result_type(counters->lower_bound(threshold), counters->end());
}

space_saving::bucket_map_ptr space_saving::get_bucket(const counter& count)
{
	auto i = counters->find(count);

	if (i != counters->cend())
		return i->second;

	auto bucket = bucket_map_ptr(new bucket_map());

	counters->insert({count, bucket});

	return bucket;
}

void space_saving::remove_entry(const counter& count, const item_id& id)
{
	auto& bucket = counters->at(count);

	bucket->erase(id);

	if (bucket->empty())
		counters->erase(count);
}
