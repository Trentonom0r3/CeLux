#include "Hsvkey_bindings.hpp"

namespace py = pybind11;

void bind_Hsvkey(py::module_ &m) {
    py::class_<Hsvkey, FilterBase, std::shared_ptr<Hsvkey>>(m, "Hsvkey")
        .def(py::init<float, float, float, float, float>(),
             py::arg("hue") = 0.00,
             py::arg("sat") = 0.00,
             py::arg("val") = 0.00,
             py::arg("similarity") = 0.01,
             py::arg("blend") = 0.00)
        .def("setHue", &Hsvkey::setHue)
        .def("getHue", &Hsvkey::getHue)
        .def("setSat", &Hsvkey::setSat)
        .def("getSat", &Hsvkey::getSat)
        .def("setVal", &Hsvkey::setVal)
        .def("getVal", &Hsvkey::getVal)
        .def("setSimilarity", &Hsvkey::setSimilarity)
        .def("getSimilarity", &Hsvkey::getSimilarity)
        .def("setBlend", &Hsvkey::setBlend)
        .def("getBlend", &Hsvkey::getBlend)
        ;
}