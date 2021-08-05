#include "space_saving.h"

using namespace std;
using namespace hh;

typedef item::id id;
typedef item::counter counter;

struct replace_item
{
	replace_item(const id& new_id, const counter& increment)
	: new_id(new_id), increment(increment)
	{}

	void operator()(item& x)
	{
		x.item_id = new_id;
		x.error = x.count;
		x.count += increment;
	}

private:
	id new_id;
	counter increment;
};

struct increase_item
{
	increase_item(const counter& increment)
	: increment(increment)
	{}

	void operator()(item& x)
	{	x.count += increment; }

private:
	counter increment;
};

space_saving::space_saving(double epsilon)
: max_items(ordered_item_map::size_type(1.0 / epsilon))
{}

void space_saving::update(const id& item_id, const counter& count)
{
	auto& i = items.get<ids>();
	auto  x = i.find(item_id);

	// update tracked item
	if (x != i.cend())
	{
		i.modify(x, increase_item(count));
		return;
	}

	// replace lowest counter
	if (items.size() == max_items)
	{
		auto& c = items.get<counts>();
		auto  y = c.begin();

		c.modify(y, replace_item(item_id, count));
		return;
	}

	// insert new item
	items.emplace(item_id, count);
}

space_saving::result_type space_saving::query(const counter& threshold) const
{
	auto& c = items.get<counts>();

	return result_type(c.lower_bound(threshold), c.end());
}

void space_saving::clear()
{
	items.get<counts>().clear();
}
