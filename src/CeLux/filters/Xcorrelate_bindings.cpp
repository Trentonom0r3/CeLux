#include "Xcorrelate_bindings.hpp"

namespace py = pybind11;

void bind_Xcorrelate(py::module_ &m) {
    py::class_<Xcorrelate, FilterBase, std::shared_ptr<Xcorrelate>>(m, "Xcorrelate")
        .def(py::init<int, int>(),
             py::arg("planes") = 7,
             py::arg("secondary") = 1)
        .def("setPlanes", &Xcorrelate::setPlanes)
        .def("getPlanes", &Xcorrelate::getPlanes)
        .def("setSecondary", &Xcorrelate::setSecondary)
        .def("getSecondary", &Xcorrelate::getSecondary)
        ;
}