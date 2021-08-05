#ifndef HHH_H
#define HHH_H

#include <utility>
#include <vector>

#include "item.h"

namespace hhh
{

constexpr static size_t HIERARCHY_SIZE = 32;
const static auto mask = [] // lambda initialization
{
	std::array<hh::item::id, HIERARCHY_SIZE + 1> masks;
	hh::item::id max = (1UL << HIERARCHY_SIZE) - 1;
	hh::item::id m = max;

	for (auto i = masks.rbegin(); i != masks.rend(); ++i)
	{
		*i = m;
		 m = m << 1 & max;
	}

	return masks;
} ();

struct label
{
	typedef size_t length;

	hh::item::id item_id;
	length len;

	label(const hh::item::id& item_id, const length& len)
	: item_id(item_id), len(len)
	{}
};

static inline bool operator==(const label& x, const label& y)
{	return x.len == y.len && x.item_id == y.item_id; }

struct label_hash
{
	typedef unsigned long hash_t;

	inline std::size_t operator()(const label& l) const
	{	return std::hash<hash_t>()(hash_t(l.len) ^ hash_t(l.item_id)); }
};

static inline hh::item::id prefix(const hh::item::id& item_id, const label::length& len)
{	return item_id & mask[len]; }

static inline label generalize(const label& l)
{
	label::length newlen = l.len - 1;

	return label(prefix(l.item_id, newlen), newlen);
}

struct hhh_item
{
	hh::item::id item_id;
	label::length len;
	hh::item::counter hi;
	hh::item::counter lo;

	inline hhh_item(const hh::item::id& item_id, const label::length& len, const hh::item::counter& hi, const hh::item::counter& lo)
	: item_id(item_id), len(len), hi(hi), lo(lo)
	{}
};

}

#endif
