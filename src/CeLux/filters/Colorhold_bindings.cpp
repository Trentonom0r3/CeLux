#include "Colorhold_bindings.hpp"

namespace py = pybind11;

void bind_Colorhold(py::module_ &m) {
    py::class_<Colorhold, FilterBase, std::shared_ptr<Colorhold>>(m, "Colorhold")
        .def(py::init<std::string, float, float>(),
             py::arg("color") = "black",
             py::arg("similarity") = 0.01,
             py::arg("blend") = 0.00)
        .def("setColor", &Colorhold::setColor)
        .def("getColor", &Colorhold::getColor)
        .def("setSimilarity", &Colorhold::setSimilarity)
        .def("getSimilarity", &Colorhold::getSimilarity)
        .def("setBlend", &Colorhold::setBlend)
        .def("getBlend", &Colorhold::getBlend)
        ;
}