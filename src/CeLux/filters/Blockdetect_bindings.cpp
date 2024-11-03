#include "Blockdetect_bindings.hpp"

namespace py = pybind11;

void bind_Blockdetect(py::module_ &m) {
    py::class_<Blockdetect, FilterBase, std::shared_ptr<Blockdetect>>(m, "Blockdetect")
        .def(py::init<int, int, int>(),
             py::arg("period_min") = 3,
             py::arg("period_max") = 24,
             py::arg("planes") = 1)
        .def("setPeriod_min", &Blockdetect::setPeriod_min)
        .def("getPeriod_min", &Blockdetect::getPeriod_min)
        .def("setPeriod_max", &Blockdetect::setPeriod_max)
        .def("getPeriod_max", &Blockdetect::getPeriod_max)
        .def("setPlanes", &Blockdetect::setPlanes)
        .def("getPlanes", &Blockdetect::getPlanes)
        ;
}