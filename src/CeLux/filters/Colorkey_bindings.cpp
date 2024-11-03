#include "Colorkey_bindings.hpp"

namespace py = pybind11;

void bind_Colorkey(py::module_ &m) {
    py::class_<Colorkey, FilterBase, std::shared_ptr<Colorkey>>(m, "Colorkey")
        .def(py::init<std::string, float, float>(),
             py::arg("color") = "black",
             py::arg("similarity") = 0.01,
             py::arg("blend") = 0.00)
        .def("setColor", &Colorkey::setColor)
        .def("getColor", &Colorkey::getColor)
        .def("setSimilarity", &Colorkey::setSimilarity)
        .def("getSimilarity", &Colorkey::getSimilarity)
        .def("setBlend", &Colorkey::setBlend)
        .def("getBlend", &Colorkey::getBlend)
        ;
}