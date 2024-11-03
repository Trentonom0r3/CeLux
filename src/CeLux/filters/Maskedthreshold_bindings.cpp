#include "Maskedthreshold_bindings.hpp"

namespace py = pybind11;

void bind_Maskedthreshold(py::module_ &m) {
    py::class_<Maskedthreshold, FilterBase, std::shared_ptr<Maskedthreshold>>(m, "Maskedthreshold")
        .def(py::init<int, int, int>(),
             py::arg("threshold") = 1,
             py::arg("planes") = 15,
             py::arg("mode") = 0)
        .def("setThreshold", &Maskedthreshold::setThreshold)
        .def("getThreshold", &Maskedthreshold::getThreshold)
        .def("setPlanes", &Maskedthreshold::setPlanes)
        .def("getPlanes", &Maskedthreshold::getPlanes)
        .def("setMode", &Maskedthreshold::setMode)
        .def("getMode", &Maskedthreshold::getMode)
        ;
}