#include "Colortemperature_bindings.hpp"

namespace py = pybind11;

void bind_Colortemperature(py::module_ &m) {
    py::class_<Colortemperature, FilterBase, std::shared_ptr<Colortemperature>>(m, "Colortemperature")
        .def(py::init<float, float, float>(),
             py::arg("temperature") = 6500.00,
             py::arg("mix") = 1.00,
             py::arg("pl") = 0.00)
        .def("setTemperature", &Colortemperature::setTemperature)
        .def("getTemperature", &Colortemperature::getTemperature)
        .def("setMix", &Colortemperature::setMix)
        .def("getMix", &Colortemperature::getMix)
        .def("setPl", &Colortemperature::setPl)
        .def("getPl", &Colortemperature::getPl)
        ;
}