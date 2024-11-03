#include "Waveform_bindings.hpp"

namespace py = pybind11;

void bind_Waveform(py::module_ &m) {
    py::class_<Waveform, FilterBase, std::shared_ptr<Waveform>>(m, "Waveform")
        .def(py::init<int, float, bool, int, int, int, int, int, float, int, int, float, float, float, int, int>(),
             py::arg("mode") = 1,
             py::arg("intensity") = 0.04,
             py::arg("mirror") = true,
             py::arg("display") = 1,
             py::arg("components") = 1,
             py::arg("envelope") = 0,
             py::arg("filter") = 0,
             py::arg("graticule") = 0,
             py::arg("opacity") = 0.75,
             py::arg("flags") = 1,
             py::arg("scale") = 0,
             py::arg("bgopacity") = 0.75,
             py::arg("tint0") = 0.00,
             py::arg("tint1") = 0.00,
             py::arg("fitmode") = 0,
             py::arg("input") = 1)
        .def("setMode", &Waveform::setMode)
        .def("getMode", &Waveform::getMode)
        .def("setIntensity", &Waveform::setIntensity)
        .def("getIntensity", &Waveform::getIntensity)
        .def("setMirror", &Waveform::setMirror)
        .def("getMirror", &Waveform::getMirror)
        .def("setDisplay", &Waveform::setDisplay)
        .def("getDisplay", &Waveform::getDisplay)
        .def("setComponents", &Waveform::setComponents)
        .def("getComponents", &Waveform::getComponents)
        .def("setEnvelope", &Waveform::setEnvelope)
        .def("getEnvelope", &Waveform::getEnvelope)
        .def("setFilter", &Waveform::setFilter)
        .def("getFilter", &Waveform::getFilter)
        .def("setGraticule", &Waveform::setGraticule)
        .def("getGraticule", &Waveform::getGraticule)
        .def("setOpacity", &Waveform::setOpacity)
        .def("getOpacity", &Waveform::getOpacity)
        .def("setFlags", &Waveform::setFlags)
        .def("getFlags", &Waveform::getFlags)
        .def("setScale", &Waveform::setScale)
        .def("getScale", &Waveform::getScale)
        .def("setBgopacity", &Waveform::setBgopacity)
        .def("getBgopacity", &Waveform::getBgopacity)
        .def("setTint0", &Waveform::setTint0)
        .def("getTint0", &Waveform::getTint0)
        .def("setTint1", &Waveform::setTint1)
        .def("getTint1", &Waveform::getTint1)
        .def("setFitmode", &Waveform::setFitmode)
        .def("getFitmode", &Waveform::getFitmode)
        .def("setInput", &Waveform::setInput)
        .def("getInput", &Waveform::getInput)
        ;
}