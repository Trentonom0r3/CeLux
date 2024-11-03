#include "Datascope_bindings.hpp"

namespace py = pybind11;

void bind_Datascope(py::module_ &m) {
    py::class_<Datascope, FilterBase, std::shared_ptr<Datascope>>(m, "Datascope")
        .def(py::init<std::pair<int, int>, int, int, int, bool, float, int, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("xOffset") = 0,
             py::arg("yOffset") = 0,
             py::arg("mode") = 0,
             py::arg("axis") = false,
             py::arg("opacity") = 0.75,
             py::arg("format") = 0,
             py::arg("components") = 15)
        .def("setSize", &Datascope::setSize)
        .def("getSize", &Datascope::getSize)
        .def("setXOffset", &Datascope::setXOffset)
        .def("getXOffset", &Datascope::getXOffset)
        .def("setYOffset", &Datascope::setYOffset)
        .def("getYOffset", &Datascope::getYOffset)
        .def("setMode", &Datascope::setMode)
        .def("getMode", &Datascope::getMode)
        .def("setAxis", &Datascope::setAxis)
        .def("getAxis", &Datascope::getAxis)
        .def("setOpacity", &Datascope::setOpacity)
        .def("getOpacity", &Datascope::getOpacity)
        .def("setFormat", &Datascope::setFormat)
        .def("getFormat", &Datascope::getFormat)
        .def("setComponents", &Datascope::setComponents)
        .def("getComponents", &Datascope::getComponents)
        ;
}