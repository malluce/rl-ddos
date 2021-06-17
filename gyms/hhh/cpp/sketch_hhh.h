#ifndef SKETCH_HHH_H
#define SKETCH_HHH_H

#include <array>
#include <memory>

#include "hhh.h"
#include "space_saving.h"

namespace hhh
{

class sketch_hhh
{
private:
	typedef std::shared_ptr<hh::space_saving> hh_ptr;
	typedef std::array<hh_ptr, HIERARCHY_SIZE + 1> sketch;
	typedef std::unordered_map<label, hh::counter, label_hash> count_map;
	typedef std::vector<hhh_entry> result_type;

public:
	sketch_hhh(const double& epsilon);

	void update(const hh::item_id& id, const hh::counter& count = 1);
	result_type query(const double& phi, const length& min_prefix_length = 0) const;

private:
	double epsilon;
	hh::counter total;
	sketch spcs;
};

}

#endif
