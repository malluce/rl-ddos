#pragma once

#include <array>
#include <iostream>
#include <vector>

#include "hhh.h"
#include "measure.h"
#include "space_saving.h"
#include "concurrentqueue.h"

namespace hhh {

class sketch
{
private:
	typedef hh::item item;
	typedef std::unordered_map<hhh::label, item::counter, hhh::label_hash> count_map;
	typedef moodycamel::ConcurrentQueue<hh::item> item_queue;
	typedef item_queue::size_t queue_size_type;
	typedef std::vector<hh::item> item_batch;
	typedef hh::space_saving space_saving;
	typedef size_t thread_id;

	constexpr static unsigned int PARALELLISM = 6;
	constexpr static queue_size_type QUEUE_SIZE = PARALELLISM * 128;
	constexpr static queue_size_type BATCH_SIZE = 10;

	static_assert(HIERARCHY_SIZE + 1 >= PARALELLISM);

public:
	typedef moodycamel::ProducerToken producer_token;
	typedef std::vector<hhh::hhh_item> result_type;

	sketch(double const& epsilon);
	~sketch();
	void flush_batch();
	void update(const item::id& item_id, const item::counter& count = 1, bool const& flush = false);
	result_type query(double const& phi, hhh::label::length const& min_prefix_length = 0);
	void clear();

private:
	void waitfor_spcs_updates_finished();
	void spcs_update(thread_id const& id);
	void spcs_loop_proxy(thread_id const& id);

	atomic_bool done;
	item::counter total;
	item_batch batch;
	item_queue update_queue;
	std::array<thread, PARALELLISM> spcs_threads;
	std::array<space_saving, HIERARCHY_SIZE + 1> spcs_instances;
	std::array<std::vector<label::length>, PARALELLISM> thread_spcs_assoc;
	std::array<producer_token, PARALELLISM> update_tokens;
	std::array<std::atomic<int>, PARALELLISM> assigned_workloads;
};

}
