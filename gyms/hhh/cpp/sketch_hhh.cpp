#include <array>
#include <memory>

#include "hhh.h"
#include "space_saving.h"
#include "sketch_hhh.h"

using namespace std;
using namespace hh;
using namespace hhh;

sketch_hhh::sketch_hhh(const double& epsilon)
: epsilon(epsilon), total(0)
{
	for (auto& s : spcs)
		s = hh_ptr(new space_saving(epsilon));
}

void sketch_hhh::update(const item_id& id, const counter& count)
{
	sketch::size_type i;

	total += count;

	#pragma omp parallel for private(i)
	for (i = 0; i < spcs.size(); ++i)
		spcs[i]->update(prefix(id, i), count);
}

// Returns id, prefix_len, f_min, f_max of HHHs for given phi and L.
typename sketch_hhh::result_type sketch_hhh::query(const double& phi, const length& min_prefix_length) const
{
	counter threshold = counter(phi * total);
	count_map current_discounts;
	count_map parent_discounts;
	result_type result;

	for (int h = spcs.size() - 1; h >= int(min_prefix_length); --h) // iterate over space saving instances, bottom-up until L
	{
		auto hh_result = spcs[h]->query(threshold);

		for (auto& i = hh_result.first; i != hh_result.second; ++i)
		{
			counter count = i->first; // f_max(e)

			for (auto& j : *i->second)
			{
				item_id id = j.second->id;
				counter d = 0; // s_e
				label l(id, h);

				auto k = current_discounts.find(l);

				if (k != current_discounts.cend())
					d = k->second;

				if (count - d >= threshold)
				{
					d = count - j.second->error;
					result.push_back(hhh_entry(id, h, count, d));
				}

				if (d != 0 && l.len != 0)
				{
					auto x = parent_discounts.insert({generalize(l), d});

					if (!x.second)
						x.first->second += d;
				}
			}
		}

		current_discounts = parent_discounts;
		parent_discounts = count_map();
	}

	return result;
}
