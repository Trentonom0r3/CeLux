#include "Thistogram_bindings.hpp"

namespace py = pybind11;

void bind_Thistogram(py::module_ &m) {
    py::class_<Thistogram, FilterBase, std::shared_ptr<Thistogram>>(m, "Thistogram")
        .def(py::init<int, int, int, int, float, bool, std::string, int>(),
             py::arg("width") = 0,
             py::arg("display_mode") = 2,
             py::arg("levels_mode") = 0,
             py::arg("components") = 7,
             py::arg("bgopacity") = 0.90,
             py::arg("envelope") = false,
             py::arg("ecolor") = "gold",
             py::arg("slide") = 1)
        .def("setWidth", &Thistogram::setWidth)
        .def("getWidth", &Thistogram::getWidth)
        .def("setDisplay_mode", &Thistogram::setDisplay_mode)
        .def("getDisplay_mode", &Thistogram::getDisplay_mode)
        .def("setLevels_mode", &Thistogram::setLevels_mode)
        .def("getLevels_mode", &Thistogram::getLevels_mode)
        .def("setComponents", &Thistogram::setComponents)
        .def("getComponents", &Thistogram::getComponents)
        .def("setBgopacity", &Thistogram::setBgopacity)
        .def("getBgopacity", &Thistogram::getBgopacity)
        .def("setEnvelope", &Thistogram::setEnvelope)
        .def("getEnvelope", &Thistogram::getEnvelope)
        .def("setEcolor", &Thistogram::setEcolor)
        .def("getEcolor", &Thistogram::getEcolor)
        .def("setSlide", &Thistogram::setSlide)
        .def("getSlide", &Thistogram::getSlide)
        ;
}