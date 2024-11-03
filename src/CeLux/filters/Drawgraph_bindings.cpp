#include "Drawgraph_bindings.hpp"

namespace py = pybind11;

void bind_Drawgraph(py::module_ &m) {
    py::class_<Drawgraph, FilterBase, std::shared_ptr<Drawgraph>>(m, "Drawgraph")
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
        .def("setM1", &Drawgraph::setM1)
        .def("getM1", &Drawgraph::getM1)
        .def("setFg1", &Drawgraph::setFg1)
        .def("getFg1", &Drawgraph::getFg1)
        .def("setM2", &Drawgraph::setM2)
        .def("getM2", &Drawgraph::getM2)
        .def("setFg2", &Drawgraph::setFg2)
        .def("getFg2", &Drawgraph::getFg2)
        .def("setM3", &Drawgraph::setM3)
        .def("getM3", &Drawgraph::getM3)
        .def("setFg3", &Drawgraph::setFg3)
        .def("getFg3", &Drawgraph::getFg3)
        .def("setM4", &Drawgraph::setM4)
        .def("getM4", &Drawgraph::getM4)
        .def("setFg4", &Drawgraph::setFg4)
        .def("getFg4", &Drawgraph::getFg4)
        .def("setBg", &Drawgraph::setBg)
        .def("getBg", &Drawgraph::getBg)
        .def("setMin", &Drawgraph::setMin)
        .def("getMin", &Drawgraph::getMin)
        .def("setMax", &Drawgraph::setMax)
        .def("getMax", &Drawgraph::getMax)
        .def("setMode", &Drawgraph::setMode)
        .def("getMode", &Drawgraph::getMode)
        .def("setSlide", &Drawgraph::setSlide)
        .def("getSlide", &Drawgraph::getSlide)
        .def("setSize", &Drawgraph::setSize)
        .def("getSize", &Drawgraph::getSize)
        .def("setRate", &Drawgraph::setRate)
        .def("getRate", &Drawgraph::getRate)
        ;
}