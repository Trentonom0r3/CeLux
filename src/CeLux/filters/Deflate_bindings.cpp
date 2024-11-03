#include "Deflate_bindings.hpp"

namespace py = pybind11;

void bind_Deflate(py::module_ &m) {
    py::class_<Deflate, FilterBase, std::shared_ptr<Deflate>>(m, "Deflate")
        .def(py::init<int, int, int, int>(),
             py::arg("threshold0") = 65535,
             py::arg("threshold1") = 65535,
             py::arg("threshold2") = 65535,
             py::arg("threshold3") = 65535)
        .def("setThreshold0", &Deflate::setThreshold0)
        .def("getThreshold0", &Deflate::getThreshold0)
        .def("setThreshold1", &Deflate::setThreshold1)
        .def("getThreshold1", &Deflate::getThreshold1)
        .def("setThreshold2", &Deflate::setThreshold2)
        .def("getThreshold2", &Deflate::getThreshold2)
        .def("setThreshold3", &Deflate::setThreshold3)
        .def("getThreshold3", &Deflate::getThreshold3)
        ;
}