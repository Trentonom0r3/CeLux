#include "Chromahold_bindings.hpp"

namespace py = pybind11;

void bind_Chromahold(py::module_ &m) {
    py::class_<Chromahold, FilterBase, std::shared_ptr<Chromahold>>(m, "Chromahold")
        .def(py::init<std::string, float, float, bool>(),
             py::arg("color") = "black",
             py::arg("similarity") = 0.01,
             py::arg("blend") = 0.00,
             py::arg("yuv") = false)
        .def("setColor", &Chromahold::setColor)
        .def("getColor", &Chromahold::getColor)
        .def("setSimilarity", &Chromahold::setSimilarity)
        .def("getSimilarity", &Chromahold::getSimilarity)
        .def("setBlend", &Chromahold::setBlend)
        .def("getBlend", &Chromahold::getBlend)
        .def("setYuv", &Chromahold::setYuv)
        .def("getYuv", &Chromahold::getYuv)
        ;
}