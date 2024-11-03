#include "Sidedata_bindings.hpp"

namespace py = pybind11;

void bind_Sidedata(py::module_ &m) {
    py::class_<Sidedata, FilterBase, std::shared_ptr<Sidedata>>(m, "Sidedata")
        .def(py::init<int, int>(),
             py::arg("mode") = 0,
             py::arg("type") = -1)
        .def("setMode", &Sidedata::setMode)
        .def("getMode", &Sidedata::getMode)
        .def("setType", &Sidedata::setType)
        .def("getType", &Sidedata::getType)
        ;
}