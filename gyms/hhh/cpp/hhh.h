#ifndef HHH_H
#define HHH_H

#include <utility>
#include <vector>

#include "item.h"

namespace hhh
{

typedef size_t length;

const static size_t HIERARCHY_SIZE = 32;
const static auto mask = [] // lambda initialization
{
	std::array<hh::item_id, HIERARCHY_SIZE + 1> masks;
	hh::item_id max = (1UL << HIERARCHY_SIZE) - 1;
	hh::item_id m = max;

	for (auto i = masks.rbegin(); i != masks.rend(); ++i)
	{
		*i = m;
		 m = m << 1 & max;
	}

	return masks;
} ();

struct label
{
	hh::item_id id;
	length len;

	label(const hh::item_id& id, const length& len)
	: id(id), len(len)
	{}
};

static inline bool operator==(const label& x, const label& y)
{	return x.len == y.len && x.id == y.id; }

struct label_hash
{
	typedef unsigned long hash_t;

	inline std::size_t operator()(const label& l) const
	{	return std::hash<hash_t>()(hash_t(l.len) ^ hash_t(l.id)); }
};

static inline hh::item_id prefix(const hh::item_id& id, const length& len)
{	return id & mask[len]; }

static inline label generalize(const label& l)
{
	length newlen = l.len - 1;

	return label(prefix(l.id, newlen), newlen);
}

struct hhh_entry
{
	hh::item_id id;
	length len;
	hh::counter hi;
	hh::counter lo;

	inline hhh_entry(const hh::item_id& id, const length& len, const hh::counter& hi, const hh::counter& lo)
	: id(id), len(len), hi(hi), lo(lo)
	{}
};

}

#endif
