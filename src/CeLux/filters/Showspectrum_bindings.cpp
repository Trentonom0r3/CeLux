#include "Showspectrum_bindings.hpp"

namespace py = pybind11;

void bind_Showspectrum(py::module_ &m) {
    py::class_<Showspectrum, FilterBase, std::shared_ptr<Showspectrum>>(m, "Showspectrum")
        .def(py::init<std::pair<int, int>, int, int, int, int, int, float, int, int, float, float, int, float, int, int, std::string, bool, float, float, float>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("slide") = 0,
             py::arg("mode") = 0,
             py::arg("color") = 0,
             py::arg("scale") = 1,
             py::arg("fscale") = 0,
             py::arg("saturation") = 1.00,
             py::arg("win_func") = 1,
             py::arg("orientation") = 0,
             py::arg("overlap") = 0.00,
             py::arg("gain") = 1.00,
             py::arg("data") = 0,
             py::arg("rotation") = 0.00,
             py::arg("start") = 0,
             py::arg("stop") = 0,
             py::arg("fps") = "auto",
             py::arg("legend") = false,
             py::arg("drange") = 120.00,
             py::arg("limit") = 0.00,
             py::arg("opacity") = 1.00)
        .def("setSize", &Showspectrum::setSize)
        .def("getSize", &Showspectrum::getSize)
        .def("setSlide", &Showspectrum::setSlide)
        .def("getSlide", &Showspectrum::getSlide)
        .def("setMode", &Showspectrum::setMode)
        .def("getMode", &Showspectrum::getMode)
        .def("setColor", &Showspectrum::setColor)
        .def("getColor", &Showspectrum::getColor)
        .def("setScale", &Showspectrum::setScale)
        .def("getScale", &Showspectrum::getScale)
        .def("setFscale", &Showspectrum::setFscale)
        .def("getFscale", &Showspectrum::getFscale)
        .def("setSaturation", &Showspectrum::setSaturation)
        .def("getSaturation", &Showspectrum::getSaturation)
        .def("setWin_func", &Showspectrum::setWin_func)
        .def("getWin_func", &Showspectrum::getWin_func)
        .def("setOrientation", &Showspectrum::setOrientation)
        .def("getOrientation", &Showspectrum::getOrientation)
        .def("setOverlap", &Showspectrum::setOverlap)
        .def("getOverlap", &Showspectrum::getOverlap)
        .def("setGain", &Showspectrum::setGain)
        .def("getGain", &Showspectrum::getGain)
        .def("setData", &Showspectrum::setData)
        .def("getData", &Showspectrum::getData)
        .def("setRotation", &Showspectrum::setRotation)
        .def("getRotation", &Showspectrum::getRotation)
        .def("setStart", &Showspectrum::setStart)
        .def("getStart", &Showspectrum::getStart)
        .def("setStop", &Showspectrum::setStop)
        .def("getStop", &Showspectrum::getStop)
        .def("setFps", &Showspectrum::setFps)
        .def("getFps", &Showspectrum::getFps)
        .def("setLegend", &Showspectrum::setLegend)
        .def("getLegend", &Showspectrum::getLegend)
        .def("setDrange", &Showspectrum::setDrange)
        .def("getDrange", &Showspectrum::getDrange)
        .def("setLimit", &Showspectrum::setLimit)
        .def("getLimit", &Showspectrum::getLimit)
        .def("setOpacity", &Showspectrum::setOpacity)
        .def("getOpacity", &Showspectrum::getOpacity)
        ;
}