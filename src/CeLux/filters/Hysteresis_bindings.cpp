#include "Hysteresis_bindings.hpp"

namespace py = pybind11;

void bind_Hysteresis(py::module_ &m) {
    py::class_<Hysteresis, FilterBase, std::shared_ptr<Hysteresis>>(m, "Hysteresis")
        .def(py::init<int, int>(),
             py::arg("planes") = 15,
             py::arg("threshold") = 0)
        .def("setPlanes", &Hysteresis::setPlanes)
        .def("getPlanes", &Hysteresis::getPlanes)
        .def("setThreshold", &Hysteresis::setThreshold)
        .def("getThreshold", &Hysteresis::getThreshold)
        ;
}