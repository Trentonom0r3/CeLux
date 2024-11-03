#include "Showvolume_bindings.hpp"

namespace py = pybind11;

void bind_Showvolume(py::module_ &m) {
    py::class_<Showvolume, FilterBase, std::shared_ptr<Showvolume>>(m, "Showvolume")
        .def(py::init<std::pair<int, int>, int, int, int, double, std::string, bool, bool, double, std::string, int, int, float, int, int>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("borderWidth") = 1,
             py::arg("channelWidth") = 400,
             py::arg("channelHeight") = 20,
             py::arg("fade") = 0.95,
             py::arg("volumeColor") = "PEAK*255+floor((1-PEAK)*255)*256+0xff000000",
             py::arg("displayChannelNames") = true,
             py::arg("displayVolume") = true,
             py::arg("dm") = 0.00,
             py::arg("dmc") = "orange",
             py::arg("orientation") = 0,
             py::arg("stepSize") = 0,
             py::arg("backgroundOpacity") = 0.00,
             py::arg("mode") = 0,
             py::arg("ds") = 0)
        .def("setRate", &Showvolume::setRate)
        .def("getRate", &Showvolume::getRate)
        .def("setBorderWidth", &Showvolume::setBorderWidth)
        .def("getBorderWidth", &Showvolume::getBorderWidth)
        .def("setChannelWidth", &Showvolume::setChannelWidth)
        .def("getChannelWidth", &Showvolume::getChannelWidth)
        .def("setChannelHeight", &Showvolume::setChannelHeight)
        .def("getChannelHeight", &Showvolume::getChannelHeight)
        .def("setFade", &Showvolume::setFade)
        .def("getFade", &Showvolume::getFade)
        .def("setVolumeColor", &Showvolume::setVolumeColor)
        .def("getVolumeColor", &Showvolume::getVolumeColor)
        .def("setDisplayChannelNames", &Showvolume::setDisplayChannelNames)
        .def("getDisplayChannelNames", &Showvolume::getDisplayChannelNames)
        .def("setDisplayVolume", &Showvolume::setDisplayVolume)
        .def("getDisplayVolume", &Showvolume::getDisplayVolume)
        .def("setDm", &Showvolume::setDm)
        .def("getDm", &Showvolume::getDm)
        .def("setDmc", &Showvolume::setDmc)
        .def("getDmc", &Showvolume::getDmc)
        .def("setOrientation", &Showvolume::setOrientation)
        .def("getOrientation", &Showvolume::getOrientation)
        .def("setStepSize", &Showvolume::setStepSize)
        .def("getStepSize", &Showvolume::getStepSize)
        .def("setBackgroundOpacity", &Showvolume::setBackgroundOpacity)
        .def("getBackgroundOpacity", &Showvolume::getBackgroundOpacity)
        .def("setMode", &Showvolume::setMode)
        .def("getMode", &Showvolume::getMode)
        .def("setDs", &Showvolume::setDs)
        .def("getDs", &Showvolume::getDs)
        ;
}