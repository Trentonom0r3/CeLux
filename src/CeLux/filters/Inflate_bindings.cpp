#include "Inflate_bindings.hpp"

namespace py = pybind11;

void bind_Inflate(py::module_ &m) {
    py::class_<Inflate, FilterBase, std::shared_ptr<Inflate>>(m, "Inflate")
        .def(py::init<int, int, int, int>(),
             py::arg("threshold0") = 65535,
             py::arg("threshold1") = 65535,
             py::arg("threshold2") = 65535,
             py::arg("threshold3") = 65535)
        .def("setThreshold0", &Inflate::setThreshold0)
        .def("getThreshold0", &Inflate::getThreshold0)
        .def("setThreshold1", &Inflate::setThreshold1)
        .def("getThreshold1", &Inflate::getThreshold1)
        .def("setThreshold2", &Inflate::setThreshold2)
        .def("getThreshold2", &Inflate::getThreshold2)
        .def("setThreshold3", &Inflate::setThreshold3)
        .def("getThreshold3", &Inflate::getThreshold3)
        ;
}