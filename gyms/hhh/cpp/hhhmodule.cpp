/*cppimport
<%
cfg['compiler_args'] = ['-std=c++17', '-O3', '-Wall', '-fPIC', '-pthread']
cfg['linker_args'] = []
cfg['sources'] = ['space_saving.cpp', 'sketch.cpp']
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sketch.h"

namespace py = pybind11;

PYBIND11_MODULE(hhhmodule, m)
{
	m.attr("HIERARCHY_SIZE") = hhh::HIERARCHY_SIZE;

	py::class_<hhh::hhh_item>(m, "HHH")
		.def(py::init<const hh::item::id&, const hhh::label::length&, const hh::item::counter&, const hh::item::counter&>())
		.def_readonly("id", &hhh::hhh_item::item_id)
		.def_readonly("len", &hhh::hhh_item::len)
		.def_readonly("hi", &hhh::hhh_item::hi)
		.def_readonly("lo", &hhh::hhh_item::lo);

	py::class_<hhh::sketch>(m, "SketchHHH")
		.def(py::init<const double&>())
		.def("update", &hhh::sketch::update,
			py::arg("id"), py::arg("count") = 1, py::arg("flush") = false)
		.def("query", &hhh::sketch::query,
			py::arg("phi"), py::arg("min_prefix_length") = 0)
		.def("clear", &hhh::sketch::clear);
}
