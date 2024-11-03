#include "Hue_bindings.hpp"

namespace py = pybind11;

void bind_Hue(py::module_ &m) {
    py::class_<Hue, FilterBase, std::shared_ptr<Hue>>(m, "Hue")
        .def(py::init<std::string, std::string, std::string, std::string>(),
             py::arg("hueAngleDegrees") = "",
             py::arg("saturation") = "1",
             py::arg("hueAngleRadians") = "",
             py::arg("brightness") = "0")
        .def("setHueAngleDegrees", &Hue::setHueAngleDegrees)
        .def("getHueAngleDegrees", &Hue::getHueAngleDegrees)
        .def("setSaturation", &Hue::setSaturation)
        .def("getSaturation", &Hue::getSaturation)
        .def("setHueAngleRadians", &Hue::setHueAngleRadians)
        .def("getHueAngleRadians", &Hue::getHueAngleRadians)
        .def("setBrightness", &Hue::setBrightness)
        .def("getBrightness", &Hue::getBrightness)
        ;
}