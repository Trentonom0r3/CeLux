#include "Overlay_bindings.hpp"

namespace py = pybind11;

void bind_Overlay(py::module_ &m) {
    py::class_<Overlay, FilterBase, std::shared_ptr<Overlay>>(m, "Overlay")
        .def(py::init<std::string, std::string, int, int, bool, int, bool, int>(),
             py::arg("x") = "0",
             py::arg("y") = "0",
             py::arg("eof_action") = 0,
             py::arg("eval") = 1,
             py::arg("shortest") = false,
             py::arg("format") = 0,
             py::arg("repeatlast") = true,
             py::arg("alpha") = 0)
        .def("setX", &Overlay::setX)
        .def("getX", &Overlay::getX)
        .def("setY", &Overlay::setY)
        .def("getY", &Overlay::getY)
        .def("setEof_action", &Overlay::setEof_action)
        .def("getEof_action", &Overlay::getEof_action)
        .def("setEval", &Overlay::setEval)
        .def("getEval", &Overlay::getEval)
        .def("setShortest", &Overlay::setShortest)
        .def("getShortest", &Overlay::getShortest)
        .def("setFormat", &Overlay::setFormat)
        .def("getFormat", &Overlay::getFormat)
        .def("setRepeatlast", &Overlay::setRepeatlast)
        .def("getRepeatlast", &Overlay::getRepeatlast)
        .def("setAlpha", &Overlay::setAlpha)
        .def("getAlpha", &Overlay::getAlpha)
        ;
}