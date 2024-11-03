#include "Histogram_bindings.hpp"

namespace py = pybind11;

void bind_Histogram(py::module_ &m) {
    py::class_<Histogram, FilterBase, std::shared_ptr<Histogram>>(m, "Histogram")
        .def(py::init<int, int, int, int, int, float, float, int>(),
             py::arg("level_height") = 200,
             py::arg("scale_height") = 12,
             py::arg("display_mode") = 2,
             py::arg("levels_mode") = 0,
             py::arg("components") = 7,
             py::arg("fgopacity") = 0.70,
             py::arg("bgopacity") = 0.50,
             py::arg("colors_mode") = 0)
        .def("setLevel_height", &Histogram::setLevel_height)
        .def("getLevel_height", &Histogram::getLevel_height)
        .def("setScale_height", &Histogram::setScale_height)
        .def("getScale_height", &Histogram::getScale_height)
        .def("setDisplay_mode", &Histogram::setDisplay_mode)
        .def("getDisplay_mode", &Histogram::getDisplay_mode)
        .def("setLevels_mode", &Histogram::setLevels_mode)
        .def("getLevels_mode", &Histogram::getLevels_mode)
        .def("setComponents", &Histogram::setComponents)
        .def("getComponents", &Histogram::getComponents)
        .def("setFgopacity", &Histogram::setFgopacity)
        .def("getFgopacity", &Histogram::getFgopacity)
        .def("setBgopacity", &Histogram::setBgopacity)
        .def("getBgopacity", &Histogram::getBgopacity)
        .def("setColors_mode", &Histogram::setColors_mode)
        .def("getColors_mode", &Histogram::getColors_mode)
        ;
}