#include "Color_bindings.hpp"

namespace py = pybind11;

void bind_Color(py::module_ &m) {
    py::class_<Color, FilterBase, std::shared_ptr<Color>>(m, "Color")
        .def(py::init<std::string, std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>>(),
             py::arg("color") = "black",
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1))
        .def("setColor", &Color::setColor)
        .def("getColor", &Color::getColor)
        .def("setSize", &Color::setSize)
        .def("getSize", &Color::getSize)
        .def("setRate", &Color::setRate)
        .def("getRate", &Color::getRate)
        .def("setDuration", &Color::setDuration)
        .def("getDuration", &Color::getDuration)
        .def("setSar", &Color::setSar)
        .def("getSar", &Color::getSar)
        ;
}