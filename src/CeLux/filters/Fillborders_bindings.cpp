#include "Fillborders_bindings.hpp"

namespace py = pybind11;

void bind_Fillborders(py::module_ &m) {
    py::class_<Fillborders, FilterBase, std::shared_ptr<Fillborders>>(m, "Fillborders")
        .def(py::init<int, int, int, int, int, std::string>(),
             py::arg("left") = 0,
             py::arg("right") = 0,
             py::arg("top") = 0,
             py::arg("bottom") = 0,
             py::arg("mode") = 0,
             py::arg("color") = "black")
        .def("setLeft", &Fillborders::setLeft)
        .def("getLeft", &Fillborders::getLeft)
        .def("setRight", &Fillborders::setRight)
        .def("getRight", &Fillborders::getRight)
        .def("setTop", &Fillborders::setTop)
        .def("getTop", &Fillborders::getTop)
        .def("setBottom", &Fillborders::setBottom)
        .def("getBottom", &Fillborders::getBottom)
        .def("setMode", &Fillborders::setMode)
        .def("getMode", &Fillborders::getMode)
        .def("setColor", &Fillborders::setColor)
        .def("getColor", &Fillborders::getColor)
        ;
}