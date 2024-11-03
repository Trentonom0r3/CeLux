#include "W3fdif_bindings.hpp"

namespace py = pybind11;

void bind_W3fdif(py::module_ &m) {
    py::class_<W3fdif, FilterBase, std::shared_ptr<W3fdif>>(m, "W3fdif")
        .def(py::init<int, int, int, int>(),
             py::arg("filter") = 1,
             py::arg("mode") = 1,
             py::arg("parity") = -1,
             py::arg("deint") = 0)
        .def("setFilter", &W3fdif::setFilter)
        .def("getFilter", &W3fdif::getFilter)
        .def("setMode", &W3fdif::setMode)
        .def("getMode", &W3fdif::getMode)
        .def("setParity", &W3fdif::setParity)
        .def("getParity", &W3fdif::getParity)
        .def("setDeint", &W3fdif::setDeint)
        .def("getDeint", &W3fdif::getDeint)
        ;
}