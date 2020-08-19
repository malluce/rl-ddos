/* g++ -Wall -O3 -std=c++11 -o pcaphist pcaphist.cpp -lboost_system -lboost_program_options -lboost_filesystem -lpcap */

#include <stdio.h>
#include <pcap/pcap.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace boost;

namespace po = boost::program_options;

typedef struct ether_header ether_hdr;
typedef const ether_header* const_ethptr;
typedef const struct iphdr* const_iphdr_ptr;
typedef uint32_t ipaddr_t;
typedef map<ipaddr_t, unsigned int> ip_counter;
typedef std::shared_ptr<ip_counter> ip_counter_ptr;

typedef struct
{
	bool use_len;
	unsigned long block_size;
	time_t time_step;
	ipaddr_t min_ip;
	ipaddr_t max_ip;
	time_t min_time;
	time_t max_time;
	size_t num_ips;
	unsigned int total_packets = 0;
	unsigned int recorded_packets = 0;
	time_t start_time = 0;
	time_t ctime = numeric_limits<time_t>::max();
	ip_counter_ptr ipctr = make_shared<ip_counter>();
	ofstream outfile;
	ipaddr_t min_observed_ip = 0xffffffff;
	ipaddr_t max_observed_ip = 0;
} loop_params;

po::variables_map cmdline(int argc, char** argv)
{
	po::options_description general("General options");
	po::options_description files("File options");
	po::options_description all("Available options");
	po::positional_options_description pos;
	po::variables_map vm;

	general.add_options()
		("help,h", "print help message")
		("block-size,b", po::value<unsigned long>()->default_value(20),
		 "size of aggregated ip blocks in bits")
		("time-step,t", po::value<time_t>()->default_value(1),
		 "size of aggregated time steps in seconds")
		("min-ip", po::value<string>()->default_value(""),
		 "minimum ip address")
		("max-ip", po::value<string>()->default_value("0xffffffff"),
		 "maximum ip address")
		("min-time", po::value<time_t>()->default_value(0),
		 "minimum time value in seconds")
		("max-time", po::value<time_t>()
			->default_value(numeric_limits<time_t>::max()),
		 "maximum time value in seconds")
		("size,s", "summarize ethernet frame length instead of packet count");

	files.add_options()
		("packet-file", po::value<string>()->required(),
		 "packet trace in pcap or pcap.gz format")
		("output-file", po::value<string>()->required(),
		 "output file for aggregated histogram data");

	pos.add("packet-file", 1);
	pos.add("output-file", 1);

	all.add(general).add(files);

	po::store(
		po::command_line_parser(argc, argv)
			.options(all)
			.positional(pos)
			.run(),
		vm
	);

	// Early exit prevents required options check.
	if (vm.count("help"))
	{
		cout << all;
		return vm;
	}

	po::notify(vm);

	return vm;
}

pcap_t* open_pcap(const string& filename)
{
	char errbuf[PCAP_ERRBUF_SIZE];
	FILE* file;

	if (algorithm::ends_with(filename, ".gz"))
	{
		string cmd = string("gzip -d -c ") + filename;
		file = popen(cmd.c_str(), "r");
	}
	else if (filename == "-")
		file = stdin;
	else
		file = fopen(filename.c_str(), "r");

	if (file == NULL)
		throw runtime_error(strerror(errno));

	auto* pcap = pcap_fopen_offline(file, errbuf);

	if (pcap == NULL)
		throw runtime_error(errbuf);

	return pcap;
}

void write_histogram(loop_params* params)
{
	auto& out = params->outfile;

	for (auto& c : *params->ipctr)
		out << (params->ctime * params->time_step)
		    << "," << c.first
		    << "," << c.second << endl;
}

void process_packet(u_char* user, const struct pcap_pkthdr* h, const u_char* pkt)
{
	auto* params = reinterpret_cast<loop_params*>(user);
	auto* eth = reinterpret_cast<const_ethptr>(pkt);

	if (eth->ether_type != htons(ETHERTYPE_IP))
		return;

	++params->total_packets;

	if (params->start_time == 0)
		params->start_time = h->ts.tv_sec;

	time_t t = (h->ts.tv_sec - params->start_time) / params->time_step;

	if (t < params->min_time || t > params->max_time)
		return;

	if (params->ctime != t)
	{
		if (params->start_time != 0)
			write_histogram(params);

		/* Reset counters but keep observed IP addresses
		 * to avoid gaps in the timeline
		 */
		for (auto& ctr : *params->ipctr)
			ctr.second = 0;

		/* Reset counter values to zero when skipping ahead */
		if (t - params->ctime > 1)
		{
			params->ctime += 1;
			write_histogram(params);
		}

		if (t - params->ctime > 1)
		{
			params->ctime = t - 1;
			write_histogram(params);
		}

		params->ctime = t;
	}

	auto ipaddr = ntohl(
		reinterpret_cast<const_iphdr_ptr>(pkt + sizeof(ether_hdr))->saddr
	);

	if (ipaddr < params->min_ip || ipaddr > params->max_ip)
		return;

	++params->recorded_packets;

	if (ipaddr < params->min_observed_ip)
		params->min_observed_ip = ipaddr;

	if (ipaddr > params->max_observed_ip)
		params->max_observed_ip = ipaddr;

	auto& bs  = params->block_size;
	auto  bin = (ipaddr >> bs) << bs;
	auto  it  = params->ipctr->find(bin);

	unsigned int inc = params->use_len ? h->len : 1;

	if (it == params->ipctr->cend())
		params->ipctr->operator[](bin) = inc;
	else
		it->second += inc;

	/* Make sure the adjacent bins exist, otherwise initialize to zero
	 * to avoid interpolation of non-existent IP addresses.
	 */

	bin -= 1 << bs;

	if (params->ipctr->find(bin) == params->ipctr->cend())
		params->ipctr->operator[](bin) = 0;

	bin += 2 << bs;

	if (params->ipctr->find(bin) == params->ipctr->cend())
		params->ipctr->operator[](bin) = 0;
}

void generate_histogram(const string& packet_file, loop_params* params)
{
	auto* pcap = open_pcap(packet_file);

	pcap_loop(pcap, 0, process_packet, reinterpret_cast<u_char*>(params));

	pcap_close(pcap);

	cout << "min observed ip: 0x" << hex << params->min_observed_ip << endl;
	cout << "max observed ip: 0x" << hex << params->max_observed_ip << endl;
	cout << "total packets:    " << dec << params->total_packets << endl;
	cout << "recorded packets: " << dec << params->recorded_packets << endl;
	cout << "skipped packets:  " << dec
	     << (params->total_packets - params->recorded_packets) << endl;
}

int main(int argc, char** argv)
{
	try
	{
		auto options = cmdline(argc, argv);

		if (options.count("help"))
			return 0;

		auto packet_file = options["packet-file"].as<string>();
		auto output_file = options["output-file"].as<string>();

		loop_params params;

		auto min_ip = strtoul(options["min-ip"].as<string>().c_str(), nullptr, 16);
		auto max_ip = strtoul(options["max-ip"].as<string>().c_str(), nullptr, 16);

		params.use_len    = options.count("size") != 0;
		params.block_size = options["block-size"].as<unsigned long>();
		params.min_ip     = min_ip;
		params.max_ip     = max_ip;
		params.time_step  = options["time-step"].as<time_t>();
		params.min_time   = options["min-time"].as<time_t>();
		params.max_time   = options["max-time"].as<time_t>();

		if (min_ip > max_ip)
			throw range_error("min IP greater than max IP.");

		params.num_ips = 1 + ((max_ip - min_ip) >> params.block_size);
		params.outfile = ofstream(output_file);

		generate_histogram(packet_file, &params);

		params.outfile.close();
	}
	catch (std::exception& e)
	{
		cerr << "Error: " << e.what() << endl;
		return 1;
	}

	return 0;
}
