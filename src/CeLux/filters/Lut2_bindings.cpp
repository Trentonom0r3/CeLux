#include "Lut2_bindings.hpp"

namespace py = pybind11;

void bind_Lut2(py::module_ &m) {
    py::class_<Lut2, FilterBase, std::shared_ptr<Lut2>>(m, "Lut2")
        .def(py::init<std::string, std::string, std::string, std::string, int>(),
             py::arg("c0") = "x",
             py::arg("c1") = "x",
             py::arg("c2") = "x",
             py::arg("c3") = "x",
             py::arg("outputDepth") = 0)
        .def("setC0", &Lut2::setC0)
        .def("getC0", &Lut2::getC0)
        .def("setC1", &Lut2::setC1)
        .def("getC1", &Lut2::getC1)
        .def("setC2", &Lut2::setC2)
        .def("getC2", &Lut2::getC2)
        .def("setC3", &Lut2::setC3)
        .def("getC3", &Lut2::getC3)
        .def("setOutputDepth", &Lut2::setOutputDepth)
        .def("getOutputDepth", &Lut2::getOutputDepth)
        ;
}