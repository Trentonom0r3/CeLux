#include "Roberts_bindings.hpp"

namespace py = pybind11;

void bind_Roberts(py::module_ &m) {
    py::class_<Roberts, FilterBase, std::shared_ptr<Roberts>>(m, "Roberts")
        .def(py::init<int, float, float>(),
             py::arg("planes") = 15,
             py::arg("scale") = 1.00,
             py::arg("delta") = 0.00)
        .def("setPlanes", &Roberts::setPlanes)
        .def("getPlanes", &Roberts::getPlanes)
        .def("setScale", &Roberts::setScale)
        .def("getScale", &Roberts::getScale)
        .def("setDelta", &Roberts::setDelta)
        .def("getDelta", &Roberts::getDelta)
        ;
}