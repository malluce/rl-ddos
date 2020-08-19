/* g++ -std=c++11 -o labelcap labelcap.cpp -O3 -lboost_system -lboost_program_options -lboost_filesystem -lpugixml -lpcap */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <arpa/inet.h>
#include <pcap/pcap.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <sys/time.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <exception>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

#include <pugixml.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/asio/ip/address.hpp>

using namespace std;
using namespace boost;
using namespace boost::asio::ip;
using namespace pugi;

namespace po = boost::program_options;

typedef struct ether_header* ethptr;
typedef struct iphdr* iphdr_ptr;
typedef struct udphdr* udphdr_ptr;
typedef struct tcphdr* tcphdr_ptr;
typedef const struct ether_header* const_ethptr;
typedef const struct iphdr* const_iphdr_ptr;
typedef const struct udphdr* const_udphdr_ptr;
typedef const struct tcphdr* const_tcphdr_ptr;
typedef std::array<char, 56> serialized_packet;
typedef struct timeval packet_time;
typedef uint32_t ipv4addr;

typedef struct {
	ofstream filter_file;
	pcap_dumper_t* pcap_dumper;
} pcap_dumpmap_entry;

typedef std::shared_ptr<pcap_dumpmap_entry> pcap_dumpmap_entry_ptr;
typedef map<unsigned long, pcap_dumpmap_entry_ptr> pcap_dumpmap;
typedef std::shared_ptr<pcap_dumpmap> pcap_dumpmap_ptr;

po::variables_map cmdline(int argc, char** argv)
{
	po::options_description general("General options");
	po::options_description label("Labelling options");
	po::options_description all("Available options");
	po::positional_options_description pos;
	po::variables_map vm;

	general.add_options()
		("help,h", "print help message")
		("split,s", "split pcap file by anomaly instead of labeling");

	label.add_options()
		("packet-file", po::value<string>()->required(),
		 "packet trace in pcap or pcap.gz format")
		("output-dir", po::value<string>()->required(),
		 "directory for produced labels and pcap trace files")
		("anomaly-files", po::value<vector<string>>()->required(),
		 "anomaly input files in XML format");

	pos.add("packet-file", 1);
	pos.add("output-dir", 1);
	pos.add("anomaly-files", -1);

	all.add(general).add(label);

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

serialized_packet serialize_packet(const struct pcap_pkthdr* h, const u_char* pkt)
{
	serialized_packet spkt = {}; // initialized with 0s

	auto* eth = reinterpret_cast<const_ethptr>(pkt);

	if (eth->ether_type != ntohs(ETHERTYPE_IP))
		return spkt;

	static struct timeval time_last = {0, 0};
	struct timeval time_delta;

	timersub(&h->ts, &time_last, &time_delta);
	time_last = h->ts;

	auto  it  = spkt.begin();
	auto* pl  = pkt + sizeof(struct ether_header);
	auto* s   = reinterpret_cast<const char*>(&time_delta);
	auto* u   = reinterpret_cast<const char*>(&time_delta);
	auto* ip  = reinterpret_cast<const_iphdr_ptr>(pl);
	auto  ihl = ip->ihl << 2;

	it = copy(s, s + sizeof(time_t), it);
	it = copy(u, u + sizeof(suseconds_t), it);
	it = copy(pl, pl + sizeof(struct iphdr), it); // truncate options

	pl += ihl;

	if (ip->protocol == IPPROTO_UDP)
	{
		auto* udp = reinterpret_cast<const_udphdr_ptr>(pl);

		copy(pl, pl + sizeof(struct udphdr), it);
	}

	if (ip->protocol == IPPROTO_TCP)
	{
		auto* tcp = reinterpret_cast<const_tcphdr_ptr>(pl);

		copy(pl, pl + sizeof(struct tcphdr), it); // truncate options
	}

	return spkt;
}

inline static
ipv4addr to_ipv4addr(const char* s)
{	return htonl(address_v4::from_string(s).to_ulong()); }

inline static
string ipv4addr_to_string(const ipv4addr& addr)
{	return address_v4(ntohl(addr)).to_string(); }

inline static
uint16_t to_port(const char* s)
{	return htons(lexical_cast<uint16_t>(s)); }

inline static
string port_to_string(const uint16_t& port)
{	return lexical_cast<string>(ntohs(port)); }

inline static
time_t string_to_seconds(const char* s)
{	return lexical_cast<time_t>(string(s)); }

inline static
suseconds_t string_to_useconds(const char* s)
{	return lexical_cast<suseconds_t>(string(s)); }

class filter
{
public:
	filter()
	{}

	void set_srcip(const uint32_t& srcip)
	{
		this->srcip = srcip;
		match_srcip = true;
	}

	void set_dstip(const uint32_t& dstip)
	{
		this->dstip = dstip;
		match_dstip = true;
	}

	void set_srcport(const uint32_t& srcport)
	{
		this->srcport = srcport;
		match_srcport = true;
	}

	void set_dstport(const uint32_t& dstport)
	{
		this->dstport = dstport;
		match_dstport = true;
	}

	bool match(const u_char* pkt) const
	{
		auto eth = reinterpret_cast<const_ethptr>(pkt);

		if (eth->ether_type != htons(ETHERTYPE_IP))
			return false;

		auto* pl    = pkt + sizeof(struct ether_header);
		auto* ip    = reinterpret_cast<const_iphdr_ptr>(pl);
		auto  ihl   = ip->ihl << 2;
		auto  proto = ip->protocol;

		if (match_srcip && srcip != ip->saddr)
			return false;

		if (match_dstip && dstip != ip->daddr)
			return false;

		if (!match_srcport && !match_dstport)
			return true;

		if (proto != IPPROTO_UDP && proto != IPPROTO_TCP)
			return false;

		pl += ihl;

		auto* udp = reinterpret_cast<const_udphdr_ptr>(pl);

		if (match_srcport && srcport != udp->source)
			return false;

		if (match_dstport && dstport != udp->dest)
			return false;

		return true;
	}

	string to_string() const
	{
		vector<string> str;

		if (match_srcip)
		{
			str.push_back("srcip");
			str.push_back(ipv4addr_to_string(srcip));
		}

		if (match_dstip)
		{
			str.push_back("dstip");
			str.push_back(ipv4addr_to_string(dstip));
		}

		if (match_srcport)
		{
			str.push_back("srcport");
			str.push_back(port_to_string(srcport));
		}

		if (match_dstport)
		{
			str.push_back("dstport");
			str.push_back(port_to_string(dstport));
		}

		return algorithm::join(str, " ");
	}

private:
	bool match_srcip = false;
	bool match_dstip = false;
	bool match_srcport = false;
	bool match_dstport = false;
	ipv4addr srcip = 0;
	ipv4addr dstip = 0;
	uint16_t srcport = 0;
	uint16_t dstport = 0;
};

typedef std::shared_ptr<filter> filter_ptr;

class anomaly
{
public:
	typedef unsigned long id_t;

	anomaly(const unsigned long& id)
	: id(id)
	{ }

	id_t get_id() const
	{	return id; }

	string get_label() const
	{	return label; }

	string get_taxonomy() const
	{	return taxonomy; }

	void set_id(const id_t& id)
	{	this->id = id; }

	void set_label(const string& label)
	{	this->label = label; }

	void set_taxonomy(const string& taxonomy)
	{	this->taxonomy = taxonomy; }

	void set_start_time(const packet_time& t)
	{	start_time = t; }

	void set_end_time(const packet_time& t)
	{	end_time = t; }

	void add_filter(filter_ptr f)
	{	filters.push_back(f); }

	vector<filter_ptr> get_filters() const
	{	return filters; }

	bool match(const struct pcap_pkthdr* h, const u_char* pkt) const
	{
		if (timercmp(&h->ts, &start_time, <=))
			return false;

		if (timercmp(&h->ts, &end_time, >=))
			return false;

		auto res = find_if(filters.cbegin(), filters.cend(),
			[pkt] (const filter_ptr& s) { return s->match(pkt); });

		return res != filters.cend();
	}

	string to_string() const
	{
		vector<string> elems;

		auto start = string(ctime(&start_time.tv_sec));
		auto end   = string(ctime(&end_time.tv_sec));

		trim_right(start);
		trim_right(end);

		elems.push_back("id " + lexical_cast<string>(id));
		elems.push_back("  label      " + label);
		elems.push_back("  taxonomy   " + taxonomy);
		elems.push_back("  start_time " + start);
		elems.push_back("  end_time   " + end);

		for (auto& f : filters)
			elems.push_back(string("  filter     ") + f->to_string());

		return algorithm::join(elems, "\n");
	}

private:
	id_t id;
	string label = string("unknown");
	string taxonomy = string("unknown");
	packet_time start_time;
	packet_time end_time;
	vector<filter_ptr> filters;
};

typedef std::shared_ptr<anomaly> anomaly_ptr;

typedef struct
{
	vector<anomaly_ptr> anomalies;
	ofstream pfile; // textual packet description
	ofstream lfile; // anomaly classification labels
	ofstream cfile; // categorical labels
	unsigned int packets = 0;
	unsigned int matches = 0;
} pcap_labeler_parameters_t;

typedef struct
{
	vector<anomaly_ptr> anomalies;
	pcap_dumpmap_ptr dumpmap = nullptr;
	unsigned int packets = 0;
	unsigned int matches = 0;
} pcap_splitter_parameters_t;

inline static
packet_time parse_time(const xml_node& node, const char* name)
{
	packet_time t;

	for (auto& attr : node.child(name).attributes())
	{
		if (string(attr.name()) == "sec")
			t.tv_sec = string_to_seconds(attr.value());
	
		if (string(attr.name()) == "usec")
			t.tv_usec = string_to_useconds(attr.value());
	}

	return t;
}

vector<anomaly_ptr> read_anomalies(const string& filename)
{
	xml_document doc;
	vector<anomaly_ptr> anomalies;
	anomaly:id_t id = 1;

	auto parse_result = doc.load_file(filename.c_str());
	auto root = doc.child("admd:annotation");

	for (auto& anom : root.children("anomaly"))
	{
		auto a = make_shared<anomaly>(anomaly(id++));

		for (auto& attr : anom.attributes())
		{
			auto name = string(attr.name());
			auto value = string(attr.value());

			if (name == "type")
				a->set_label(value);
			else if (name == "value")
			{
				vector<string> entries;

				split(entries, value, is_any_of(","));

				a->set_taxonomy(entries.back());
			}
		}

		for (auto& filter_spec : anom.child("slice").children("filter"))
		{
			auto f = make_shared<filter>(filter());

			for (auto& attr : filter_spec.attributes())
			{
				auto name = string(attr.name());

				if (name == "src_ip")
					f->set_srcip(to_ipv4addr(attr.value()));

				if (name == "dst_ip")
					f->set_dstip(to_ipv4addr(attr.value()));

				if (name == "src_port")
					f->set_srcport(to_port(attr.value()));

				if (name == "dst_port")
					f->set_dstport(to_port(attr.value()));
			}

			a->add_filter(f);
		}

		a->set_start_time(parse_time(anom, "from"));
		a->set_end_time(parse_time(anom, "to"));

		anomalies.push_back(a);

		cout << a->to_string() << endl;
	}

	return anomalies;
}

vector<anomaly_ptr> read_anomaly_files(const vector<string>& files)
{
	auto anomalies = vector<anomaly_ptr>();

	for (auto& file : files)
	{
		auto a = read_anomalies(file);
		anomalies.insert(anomalies.end(), a.begin(), a.end());
	}

	return anomalies;
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

pcap_dumper_t* open_pcap_dump(const string& filename, pcap_t* pcap)
{
	string cmd = string("gzip > ") + filename + string(".pcap.gz");

	auto dump = pcap_dump_fopen(pcap, popen(cmd.c_str(), "w"));

	if (dump == NULL)
		throw runtime_error(pcap_geterr(pcap));

	return dump;
}

pcap_dumpmap_ptr create_dumpmap(const vector<anomaly_ptr>& anomalies, const string& outdir, pcap_t* pcap)
{
	auto dumpmap = make_shared<pcap_dumpmap>(pcap_dumpmap());

	// Output file for non-anomalous packets
	auto entry = make_shared<pcap_dumpmap_entry>(pcap_dumpmap_entry());
	auto fname = outdir + "/normal";

	entry->filter_file = ofstream(fname + ".txt");
	entry->pcap_dumper = open_pcap_dump(fname , pcap);

	dumpmap->insert({0, entry});

	// Output files for anomalous traffic
	for (auto& a : anomalies)
	{
		auto id = a->get_id();

		if (dumpmap->find(id) != dumpmap->cend())
			continue;

		entry = make_shared<pcap_dumpmap_entry>(pcap_dumpmap_entry());
		fname =   outdir + "/" + a->get_label() + "-" + to_string(id)
		        + "-" + a->get_taxonomy();

		entry->filter_file = ofstream(fname + ".txt");
		entry->pcap_dumper = open_pcap_dump(fname , pcap);

		dumpmap->insert({id, entry});
	}

	return dumpmap;
}

void pcap_labeler(u_char* user, const struct pcap_pkthdr* h, const u_char* pkt)
{
	auto* params = reinterpret_cast<pcap_labeler_parameters_t*>(user);
	auto* eth = reinterpret_cast<const_ethptr>(pkt);
	auto& anomalies = params->anomalies;
	auto& lfile = params->lfile;
	auto& cfile = params->cfile;

	if (eth->ether_type != htons(ETHERTYPE_IP))
		return;

	++params->packets;

	auto spkt = serialize_packet(h, pkt);

	params->pfile.write(spkt.begin(), spkt.size());

	auto it = find_if(anomalies.cbegin(), anomalies.cend(),
		[h, pkt] (const anomaly_ptr& x) { return x->match(h, pkt); });

	if (it == anomalies.cend())
	{
		lfile << 0 << endl;
		cfile << 0 << endl;
		return;
	}

	++params->matches;

	lfile << 1 << endl;
	cfile << (*it)->get_id() << endl;
}

void pcap_splitter(u_char* user, const struct pcap_pkthdr* h, const u_char* pkt)
{
	auto* params = reinterpret_cast<pcap_splitter_parameters_t*>(user);
	auto* eth = reinterpret_cast<const_ethptr>(pkt);
	auto& anomalies = params->anomalies;
	auto& dumpmap = params->dumpmap;

	if (eth->ether_type != htons(ETHERTYPE_IP))
		return;

	++params->packets;

	auto it = find_if(anomalies.cbegin(), anomalies.cend(),
		[h, pkt] (const anomaly_ptr& x) { return x->match(h, pkt); });

	// default to id 0 for normal traffic
	unsigned long id = 0;

	if (it != anomalies.cend())
	{
		++params->matches;
		id = (*it)->get_id();
	}

	auto dentry = dumpmap->at(id);

	pcap_dump(reinterpret_cast<u_char*>(dentry->pcap_dumper), h, pkt);
}

void perform_labeling(const string& packet_file, const string& output_dir, const vector<string>& anomaly_files)
{
	pcap_labeler_parameters_t params;

	auto* pcap       = open_pcap(packet_file);
	params.anomalies = read_anomaly_files(anomaly_files);

	cout << "anomalies: " << params.anomalies.size() << endl;

	params.pfile = ofstream(output_dir + "/" + "packets.bin");
	params.lfile = ofstream(output_dir + "/" + "labels.txt");
	params.cfile = ofstream(output_dir + "/" + "categoricals.txt");

	pcap_loop(pcap, 0, pcap_labeler, reinterpret_cast<u_char*>(&params));

	cout << "total: " << params.packets << endl;
	cout << "match: " << params.matches << endl;

	pcap_close(pcap);
	params.pfile.close();
	params.lfile.close();
	params.cfile.close();
}

void perform_splitting(const string& packet_file, const string& output_dir, const vector<string>& anomaly_files)
{
	pcap_splitter_parameters_t params;

	auto* pcap       = open_pcap(packet_file);
	params.anomalies = read_anomaly_files(anomaly_files);
	params.dumpmap   = create_dumpmap(params.anomalies, output_dir, pcap);

	cout << "anomalies: " << params.anomalies.size() << endl;

	pcap_loop(pcap, 0, pcap_splitter, reinterpret_cast<u_char*>(&params));

	for (auto& anomaly : params.anomalies)
	for (auto& filter : anomaly->get_filters())
		params.dumpmap->at(anomaly->get_id())->filter_file
			<< filter->to_string()
			<< endl;

	cout << "total: " << params.packets << endl;
	cout << "match: " << params.matches << endl;

	pcap_close(pcap);

	for (auto& dump : *params.dumpmap)
	{
		auto dentry = dump.second;
		pcap_dump_close(dentry->pcap_dumper);
		dentry->filter_file.close();
	}
}

int main(int argc, char** argv)
{
	char errbuf[PCAP_ERRBUF_SIZE];

	try
	{
		auto options = cmdline(argc, argv);

		if (options.count("help"))
			return 0;

		auto pkt  = options["packet-file"].as<string>();
		auto out  = options["output-dir"].as<string>();
		auto anom = options["anomaly-files"].as<vector<string>>();

		filesystem::create_directory(filesystem::path(out));

		if (options.count("split"))
			perform_splitting(pkt, out, anom);
		else
			perform_labeling(pkt, out, anom);
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}
