#include "Adrawgraph_bindings.hpp"

namespace py = pybind11;

void bind_Adrawgraph(py::module_ &m) {
    py::class_<Adrawgraph, FilterBase, std::shared_ptr<Adrawgraph>>(m, "Adrawgraph")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, float, float, int, int, std::pair<int, int>, std::pair<int, int>>(),
             py::arg("m1") = "",
             py::arg("fg1") = "0xffff0000",
             py::arg("m2") = "",
             py::arg("fg2") = "0xff00ff00",
             py::arg("m3") = "",
             py::arg("fg3") = "0xffff00ff",
             py::arg("m4") = "",
             py::arg("fg4") = "0xffffff00",
             py::arg("bg") = "white",
             py::arg("min") = -1.00,
             py::arg("max") = 1.00,
             py::arg("mode") = 2,
             py::arg("slide") = 0,
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1))
        .def("setM1", &Adrawgraph::setM1)
        .def("getM1", &Adrawgraph::getM1)
        .def("setFg1", &Adrawgraph::setFg1)
        .def("getFg1", &Adrawgraph::getFg1)
        .def("setM2", &Adrawgraph::setM2)
        .def("getM2", &Adrawgraph::getM2)
        .def("setFg2", &Adrawgraph::setFg2)
        .def("getFg2", &Adrawgraph::getFg2)
        .def("setM3", &Adrawgraph::setM3)
        .def("getM3", &Adrawgraph::getM3)
        .def("setFg3", &Adrawgraph::setFg3)
        .def("getFg3", &Adrawgraph::getFg3)
        .def("setM4", &Adrawgraph::setM4)
        .def("getM4", &Adrawgraph::getM4)
        .def("setFg4", &Adrawgraph::setFg4)
        .def("getFg4", &Adrawgraph::getFg4)
        .def("setBg", &Adrawgraph::setBg)
        .def("getBg", &Adrawgraph::getBg)
        .def("setMin", &Adrawgraph::setMin)
        .def("getMin", &Adrawgraph::getMin)
        .def("setMax", &Adrawgraph::setMax)
        .def("getMax", &Adrawgraph::getMax)
        .def("setMode", &Adrawgraph::setMode)
        .def("getMode", &Adrawgraph::getMode)
        .def("setSlide", &Adrawgraph::setSlide)
        .def("getSlide", &Adrawgraph::getSlide)
        .def("setSize", &Adrawgraph::setSize)
        .def("getSize", &Adrawgraph::getSize)
        .def("setRate", &Adrawgraph::setRate)
        .def("getRate", &Adrawgraph::getRate)
        ;
}