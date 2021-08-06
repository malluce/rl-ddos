#include <iostream>
#include <utility>

#include "sketch.h"

using namespace hhh;
using namespace std;

template<size_t... ix>
std::array<hh::space_saving, sizeof...(ix)> make_space_saving_instances(double epsilon, index_sequence<ix...>)
{   return { ((void)ix, hh::space_saving(epsilon))... }; }

template<size_t... ix>
std::array<sketch::producer_token, sizeof...(ix)> make_update_tokens(sketch::item_queue& queue, index_sequence<ix...>)
{   return { ((void)ix, sketch::producer_token(queue))... }; }

sketch::sketch(double const& epsilon)
: done(false), total(0), batch(),
  update_queue(QUEUE_SIZE, PARALELLISM, 0),
  spcs_instances(make_space_saving_instances(epsilon, make_index_sequence<HIERARCHY_SIZE + 1>())),
  update_tokens(make_update_tokens(update_queue, make_index_sequence<PARALELLISM>()))
{
	auto max_level = HIERARCHY_SIZE + 1;

	for (auto level = label::length(0); level < max_level;)
		for (auto it = thread_spcs_assoc.begin(); it != thread_spcs_assoc.end() && level < max_level; ++it)
			it->push_back(level++);
	// start processing
	for (auto id = thread_id(0); id != PARALELLISM; ++id) {
		assigned_workloads[id] = 0;
		spcs_threads[id] = thread(&sketch::spcs_loop_proxy, this, id);
	}
}

sketch::~sketch()
{
	done = true;
	for (auto& t: spcs_threads)
		if (t.joinable())
			t.join();
}

void sketch::flush_batch()
{
	for (auto id = thread_id(0); id < PARALELLISM; ++id) {
		auto batch_size = batch.size();
		assigned_workloads[id].fetch_add(batch_size, std::memory_order_release);
		while (!update_queue.try_enqueue_bulk(update_tokens[id], batch.begin(), batch_size));
	}
	batch = item_batch();
}

void sketch::update(const item::id& item_id, const item::counter& count, bool const& flush)
{
	total += count;
	batch.emplace_back(item(item_id, count));
	if (flush || BATCH_SIZE == batch.size())
		flush_batch();
}

void sketch::spcs_update(thread_id const& id)
{
	item_batch batch;

	while (update_queue.try_dequeue_bulk_from_producer(update_tokens[id], back_inserter(batch), BATCH_SIZE)) {
		for (auto& level : thread_spcs_assoc[id]) {
			for (auto& i : batch) {
				spcs_instances[level].update(prefix(i.item_id, level), i.count);
			}
		}
		assigned_workloads[id].fetch_sub(batch.size(), std::memory_order_release);
		batch.clear();
	}
}

void sketch::spcs_loop_proxy(thread_id const& id)
{
	while (!done) spcs_update(id);
    spcs_update(id);
}

void sketch::waitfor_spcs_updates_finished()
{
	for (auto& load : assigned_workloads)
		while (load.load(std::memory_order_release) != 0);
}

sketch::result_type sketch::query(double const& phi, label::length const& min_prefix_length)
{
	auto threshold = item::counter(phi * total);
	count_map discounts;
	sketch::result_type result;

    flush_batch();
	waitfor_spcs_updates_finished();
	for (auto h = int(HIERARCHY_SIZE); h >= int(min_prefix_length); --h)
	{
		auto [begin, end] = spcs_instances[h].query(threshold);
		for (auto& hh = begin; hh != end; ++hh) {
			auto& item_id = hh->item_id;
			auto& count = hh->count;
			auto  discount = item::counter(0);
			label l(item_id, h);

			auto it = discounts.find(l);
			if (it != discounts.cend())
				discount = it->second;
			if (count - discount >= threshold) {
				discount = count - hh->error;
				result.push_back(hhh_item(item_id, h, count, discount));
				discounts[l] = discount;
			}
		}

		// upward propagation of discounts
		count_map t;
		for (auto& d : discounts)
			t[generalize(d.first)] += d.second;
		discounts = t;
	}
	return result;
}

void sketch::clear()
{
	total = item::counter(0);
	for (auto& s : spcs_instances)
		s.clear();
}