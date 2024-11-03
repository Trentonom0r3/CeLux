#include "Tlut2_bindings.hpp"

namespace py = pybind11;

void bind_Tlut2(py::module_ &m) {
    py::class_<Tlut2, FilterBase, std::shared_ptr<Tlut2>>(m, "Tlut2")
        .def(py::init<std::string, std::string, std::string, std::string>(),
             py::arg("c0") = "x",
             py::arg("c1") = "x",
             py::arg("c2") = "x",
             py::arg("c3") = "x")
        .def("setC0", &Tlut2::setC0)
        .def("getC0", &Tlut2::getC0)
        .def("setC1", &Tlut2::setC1)
        .def("getC1", &Tlut2::getC1)
        .def("setC2", &Tlut2::setC2)
        .def("getC2", &Tlut2::getC2)
        .def("setC3", &Tlut2::setC3)
        .def("getC3", &Tlut2::getC3)
        ;
}