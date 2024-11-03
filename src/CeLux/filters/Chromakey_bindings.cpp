#include "Chromakey_bindings.hpp"

namespace py = pybind11;

void bind_Chromakey(py::module_ &m) {
    py::class_<Chromakey, FilterBase, std::shared_ptr<Chromakey>>(m, "Chromakey")
        .def(py::init<std::string, float, float, bool>(),
             py::arg("color") = "black",
             py::arg("similarity") = 0.01,
             py::arg("blend") = 0.00,
             py::arg("yuv") = false)
        .def("setColor", &Chromakey::setColor)
        .def("getColor", &Chromakey::getColor)
        .def("setSimilarity", &Chromakey::setSimilarity)
        .def("getSimilarity", &Chromakey::getSimilarity)
        .def("setBlend", &Chromakey::setBlend)
        .def("getBlend", &Chromakey::getBlend)
        .def("setYuv", &Chromakey::setYuv)
        .def("getYuv", &Chromakey::getYuv)
        ;
}