#include "Showspectrumpic_bindings.hpp"

namespace py = pybind11;

void bind_Showspectrumpic(py::module_ &m) {
    py::class_<Showspectrumpic, FilterBase, std::shared_ptr<Showspectrumpic>>(m, "Showspectrumpic")
        .def(py::init<std::pair<int, int>, int, int, int, int, float, int, int, float, bool, float, int, int, float, float, float>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("mode") = 0,
             py::arg("color") = 1,
             py::arg("scale") = 3,
             py::arg("fscale") = 0,
             py::arg("saturation") = 1.00,
             py::arg("win_func") = 1,
             py::arg("orientation") = 0,
             py::arg("gain") = 1.00,
             py::arg("legend") = true,
             py::arg("rotation") = 0.00,
             py::arg("start") = 0,
             py::arg("stop") = 0,
             py::arg("drange") = 120.00,
             py::arg("limit") = 0.00,
             py::arg("opacity") = 1.00)
        .def("setSize", &Showspectrumpic::setSize)
        .def("getSize", &Showspectrumpic::getSize)
        .def("setMode", &Showspectrumpic::setMode)
        .def("getMode", &Showspectrumpic::getMode)
        .def("setColor", &Showspectrumpic::setColor)
        .def("getColor", &Showspectrumpic::getColor)
        .def("setScale", &Showspectrumpic::setScale)
        .def("getScale", &Showspectrumpic::getScale)
        .def("setFscale", &Showspectrumpic::setFscale)
        .def("getFscale", &Showspectrumpic::getFscale)
        .def("setSaturation", &Showspectrumpic::setSaturation)
        .def("getSaturation", &Showspectrumpic::getSaturation)
        .def("setWin_func", &Showspectrumpic::setWin_func)
        .def("getWin_func", &Showspectrumpic::getWin_func)
        .def("setOrientation", &Showspectrumpic::setOrientation)
        .def("getOrientation", &Showspectrumpic::getOrientation)
        .def("setGain", &Showspectrumpic::setGain)
        .def("getGain", &Showspectrumpic::getGain)
        .def("setLegend", &Showspectrumpic::setLegend)
        .def("getLegend", &Showspectrumpic::getLegend)
        .def("setRotation", &Showspectrumpic::setRotation)
        .def("getRotation", &Showspectrumpic::getRotation)
        .def("setStart", &Showspectrumpic::setStart)
        .def("getStart", &Showspectrumpic::getStart)
        .def("setStop", &Showspectrumpic::setStop)
        .def("getStop", &Showspectrumpic::getStop)
        .def("setDrange", &Showspectrumpic::setDrange)
        .def("getDrange", &Showspectrumpic::getDrange)
        .def("setLimit", &Showspectrumpic::setLimit)
        .def("getLimit", &Showspectrumpic::getLimit)
        .def("setOpacity", &Showspectrumpic::setOpacity)
        .def("getOpacity", &Showspectrumpic::getOpacity)
        ;
}