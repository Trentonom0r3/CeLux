#include "Lut_bindings.hpp"

namespace py = pybind11;

void bind_Lut(py::module_ &m) {
    py::class_<Lut, FilterBase, std::shared_ptr<Lut>>(m, "Lut")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string>(),
             py::arg("c0") = "clipval",
             py::arg("c1") = "clipval",
             py::arg("c2") = "clipval",
             py::arg("c3") = "clipval",
             py::arg("y") = "clipval",
             py::arg("u") = "clipval",
             py::arg("v") = "clipval",
             py::arg("r") = "clipval",
             py::arg("g") = "clipval",
             py::arg("b") = "clipval",
             py::arg("a") = "clipval")
        .def("setC0", &Lut::setC0)
        .def("getC0", &Lut::getC0)
        .def("setC1", &Lut::setC1)
        .def("getC1", &Lut::getC1)
        .def("setC2", &Lut::setC2)
        .def("getC2", &Lut::getC2)
        .def("setC3", &Lut::setC3)
        .def("getC3", &Lut::getC3)
        .def("setY", &Lut::setY)
        .def("getY", &Lut::getY)
        .def("setU", &Lut::setU)
        .def("getU", &Lut::getU)
        .def("setV", &Lut::setV)
        .def("getV", &Lut::getV)
        .def("setR", &Lut::setR)
        .def("getR", &Lut::getR)
        .def("setG", &Lut::setG)
        .def("getG", &Lut::getG)
        .def("setB", &Lut::setB)
        .def("getB", &Lut::getB)
        .def("setA", &Lut::setA)
        .def("getA", &Lut::getA)
        ;
}