#include "Swaprect_bindings.hpp"

namespace py = pybind11;

void bind_Swaprect(py::module_ &m) {
    py::class_<Swaprect, FilterBase, std::shared_ptr<Swaprect>>(m, "Swaprect")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string>(),
             py::arg("rectWidth") = "w/2",
             py::arg("rectHeight") = "h/2",
             py::arg("x1") = "w/2",
             py::arg("y1") = "h/2",
             py::arg("x2") = "0",
             py::arg("y2") = "0")
        .def("setRectWidth", &Swaprect::setRectWidth)
        .def("getRectWidth", &Swaprect::getRectWidth)
        .def("setRectHeight", &Swaprect::setRectHeight)
        .def("getRectHeight", &Swaprect::getRectHeight)
        .def("setX1", &Swaprect::setX1)
        .def("getX1", &Swaprect::getX1)
        .def("setY1", &Swaprect::setY1)
        .def("getY1", &Swaprect::getY1)
        .def("setX2", &Swaprect::setX2)
        .def("getX2", &Swaprect::getX2)
        .def("setY2", &Swaprect::setY2)
        .def("getY2", &Swaprect::getY2)
        ;
}