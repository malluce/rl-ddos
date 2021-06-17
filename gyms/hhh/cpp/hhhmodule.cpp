/*cppimport
<%
cfg['compiler_args'] = ['-std=c++11', '-O3', '-Wall', '-fPIC', '-fopenmp']
cfg['linker_args'] = ['-fopenmp']
cfg['sources'] = ['space_saving.cpp', 'sketch_hhh.cpp']
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sketch_hhh.h"

namespace py = pybind11;

PYBIND11_MODULE(hhhmodule, m)
{
	m.attr("HIERARCHY_SIZE") = hhh::HIERARCHY_SIZE;

	py::class_<hhh::hhh_entry>(m, "HHH")
		.def(py::init<const hh::item_id&, const hhh::length&, const hh::counter&, const hh::counter&>())
		.def_readonly("id", &hhh::hhh_entry::id)
		.def_readonly("len", &hhh::hhh_entry::len)
		.def_readonly("hi", &hhh::hhh_entry::hi)
		.def_readonly("lo", &hhh::hhh_entry::lo);

	py::class_<hhh::sketch_hhh>(m, "SketchHHH")
		.def(py::init<const double&>())
		.def("update", &hhh::sketch_hhh::update,
			py::arg("id"), py::arg("count") = 1)
		.def("query", &hhh::sketch_hhh::query,
			py::arg("phi"), py::arg("min_prefix_length") = 0);
}
