#include "Threshold_bindings.hpp"

namespace py = pybind11;

void bind_Threshold(py::module_ &m) {
    py::class_<Threshold, FilterBase, std::shared_ptr<Threshold>>(m, "Threshold")
        .def(py::init<int>(),
             py::arg("planes") = 15)
        .def("setPlanes", &Threshold::setPlanes)
        .def("getPlanes", &Threshold::getPlanes)
        ;
}