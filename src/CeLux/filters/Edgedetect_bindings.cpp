#include "Edgedetect_bindings.hpp"

namespace py = pybind11;

void bind_Edgedetect(py::module_ &m) {
    py::class_<Edgedetect, FilterBase, std::shared_ptr<Edgedetect>>(m, "Edgedetect")
        .def(py::init<double, double, int, int>(),
             py::arg("high") = 0.20,
             py::arg("low") = 0.08,
             py::arg("mode") = 0,
             py::arg("planes") = 7)
        .def("setHigh", &Edgedetect::setHigh)
        .def("getHigh", &Edgedetect::getHigh)
        .def("setLow", &Edgedetect::setLow)
        .def("getLow", &Edgedetect::getLow)
        .def("setMode", &Edgedetect::setMode)
        .def("getMode", &Edgedetect::getMode)
        .def("setPlanes", &Edgedetect::setPlanes)
        .def("getPlanes", &Edgedetect::getPlanes)
        ;
}