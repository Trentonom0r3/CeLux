#include "Lutrgb_bindings.hpp"

namespace py = pybind11;

void bind_Lutrgb(py::module_ &m) {
    py::class_<Lutrgb, FilterBase, std::shared_ptr<Lutrgb>>(m, "Lutrgb")
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
        .def("setC0", &Lutrgb::setC0)
        .def("getC0", &Lutrgb::getC0)
        .def("setC1", &Lutrgb::setC1)
        .def("getC1", &Lutrgb::getC1)
        .def("setC2", &Lutrgb::setC2)
        .def("getC2", &Lutrgb::getC2)
        .def("setC3", &Lutrgb::setC3)
        .def("getC3", &Lutrgb::getC3)
        .def("setY", &Lutrgb::setY)
        .def("getY", &Lutrgb::getY)
        .def("setU", &Lutrgb::setU)
        .def("getU", &Lutrgb::getU)
        .def("setV", &Lutrgb::setV)
        .def("getV", &Lutrgb::getV)
        .def("setR", &Lutrgb::setR)
        .def("getR", &Lutrgb::getR)
        .def("setG", &Lutrgb::setG)
        .def("getG", &Lutrgb::getG)
        .def("setB", &Lutrgb::setB)
        .def("getB", &Lutrgb::getB)
        .def("setA", &Lutrgb::setA)
        .def("getA", &Lutrgb::getA)
        ;
}