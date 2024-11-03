#include "Pseudocolor_bindings.hpp"

namespace py = pybind11;

void bind_Pseudocolor(py::module_ &m) {
    py::class_<Pseudocolor, FilterBase, std::shared_ptr<Pseudocolor>>(m, "Pseudocolor")
        .def(py::init<std::string, std::string, std::string, std::string, int, int, float>(),
             py::arg("c0") = "val",
             py::arg("c1") = "val",
             py::arg("c2") = "val",
             py::arg("c3") = "val",
             py::arg("index") = 0,
             py::arg("preset") = -1,
             py::arg("opacity") = 1.00)
        .def("setC0", &Pseudocolor::setC0)
        .def("getC0", &Pseudocolor::getC0)
        .def("setC1", &Pseudocolor::setC1)
        .def("getC1", &Pseudocolor::getC1)
        .def("setC2", &Pseudocolor::setC2)
        .def("getC2", &Pseudocolor::getC2)
        .def("setC3", &Pseudocolor::setC3)
        .def("getC3", &Pseudocolor::getC3)
        .def("setIndex", &Pseudocolor::setIndex)
        .def("getIndex", &Pseudocolor::getIndex)
        .def("setPreset", &Pseudocolor::setPreset)
        .def("getPreset", &Pseudocolor::getPreset)
        .def("setOpacity", &Pseudocolor::setOpacity)
        .def("getOpacity", &Pseudocolor::getOpacity)
        ;
}