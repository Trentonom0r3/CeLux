#include "Lutyuv_bindings.hpp"

namespace py = pybind11;

void bind_Lutyuv(py::module_ &m) {
    py::class_<Lutyuv, FilterBase, std::shared_ptr<Lutyuv>>(m, "Lutyuv")
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
        .def("setC0", &Lutyuv::setC0)
        .def("getC0", &Lutyuv::getC0)
        .def("setC1", &Lutyuv::setC1)
        .def("getC1", &Lutyuv::getC1)
        .def("setC2", &Lutyuv::setC2)
        .def("getC2", &Lutyuv::getC2)
        .def("setC3", &Lutyuv::setC3)
        .def("getC3", &Lutyuv::getC3)
        .def("setY", &Lutyuv::setY)
        .def("getY", &Lutyuv::getY)
        .def("setU", &Lutyuv::setU)
        .def("getU", &Lutyuv::getU)
        .def("setV", &Lutyuv::setV)
        .def("getV", &Lutyuv::getV)
        .def("setR", &Lutyuv::setR)
        .def("getR", &Lutyuv::getR)
        .def("setG", &Lutyuv::setG)
        .def("getG", &Lutyuv::getG)
        .def("setB", &Lutyuv::setB)
        .def("getB", &Lutyuv::getB)
        .def("setA", &Lutyuv::setA)
        .def("getA", &Lutyuv::getA)
        ;
}