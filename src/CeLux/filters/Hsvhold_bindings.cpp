#include "Hsvhold_bindings.hpp"

namespace py = pybind11;

void bind_Hsvhold(py::module_ &m) {
    py::class_<Hsvhold, FilterBase, std::shared_ptr<Hsvhold>>(m, "Hsvhold")
        .def(py::init<float, float, float, float, float>(),
             py::arg("hue") = 0.00,
             py::arg("sat") = 0.00,
             py::arg("val") = 0.00,
             py::arg("similarity") = 0.01,
             py::arg("blend") = 0.00)
        .def("setHue", &Hsvhold::setHue)
        .def("getHue", &Hsvhold::getHue)
        .def("setSat", &Hsvhold::setSat)
        .def("getSat", &Hsvhold::getSat)
        .def("setVal", &Hsvhold::setVal)
        .def("getVal", &Hsvhold::getVal)
        .def("setSimilarity", &Hsvhold::setSimilarity)
        .def("getSimilarity", &Hsvhold::getSimilarity)
        .def("setBlend", &Hsvhold::setBlend)
        .def("getBlend", &Hsvhold::getBlend)
        ;
}