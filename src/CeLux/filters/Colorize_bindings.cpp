#include "Colorize_bindings.hpp"

namespace py = pybind11;

void bind_Colorize(py::module_ &m) {
    py::class_<Colorize, FilterBase, std::shared_ptr<Colorize>>(m, "Colorize")
        .def(py::init<float, float, float, float>(),
             py::arg("hue") = 0.00,
             py::arg("saturation") = 0.50,
             py::arg("lightness") = 0.50,
             py::arg("mix") = 1.00)
        .def("setHue", &Colorize::setHue)
        .def("getHue", &Colorize::getHue)
        .def("setSaturation", &Colorize::setSaturation)
        .def("getSaturation", &Colorize::getSaturation)
        .def("setLightness", &Colorize::setLightness)
        .def("getLightness", &Colorize::getLightness)
        .def("setMix", &Colorize::setMix)
        .def("getMix", &Colorize::getMix)
        ;
}