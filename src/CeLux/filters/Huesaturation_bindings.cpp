#include "Huesaturation_bindings.hpp"

namespace py = pybind11;

void bind_Huesaturation(py::module_ &m) {
    py::class_<Huesaturation, FilterBase, std::shared_ptr<Huesaturation>>(m, "Huesaturation")
        .def(py::init<float, float, float, int, float, float, float, float, bool>(),
             py::arg("hue") = 0.00,
             py::arg("saturation") = 0.00,
             py::arg("intensity") = 0.00,
             py::arg("colors") = 63,
             py::arg("strength") = 1.00,
             py::arg("rw") = 0.33,
             py::arg("gw") = 0.33,
             py::arg("bw") = 0.33,
             py::arg("lightness") = false)
        .def("setHue", &Huesaturation::setHue)
        .def("getHue", &Huesaturation::getHue)
        .def("setSaturation", &Huesaturation::setSaturation)
        .def("getSaturation", &Huesaturation::getSaturation)
        .def("setIntensity", &Huesaturation::setIntensity)
        .def("getIntensity", &Huesaturation::getIntensity)
        .def("setColors", &Huesaturation::setColors)
        .def("getColors", &Huesaturation::getColors)
        .def("setStrength", &Huesaturation::setStrength)
        .def("getStrength", &Huesaturation::getStrength)
        .def("setRw", &Huesaturation::setRw)
        .def("getRw", &Huesaturation::getRw)
        .def("setGw", &Huesaturation::setGw)
        .def("getGw", &Huesaturation::getGw)
        .def("setBw", &Huesaturation::setBw)
        .def("getBw", &Huesaturation::getBw)
        .def("setLightness", &Huesaturation::setLightness)
        .def("getLightness", &Huesaturation::getLightness)
        ;
}