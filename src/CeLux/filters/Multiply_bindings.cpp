#include "Multiply_bindings.hpp"

namespace py = pybind11;

void bind_Multiply(py::module_ &m) {
    py::class_<Multiply, FilterBase, std::shared_ptr<Multiply>>(m, "Multiply")
        .def(py::init<float, float, int>(),
             py::arg("scale") = 1.00,
             py::arg("offset") = 0.50,
             py::arg("planes") = 15)
        .def("setScale", &Multiply::setScale)
        .def("getScale", &Multiply::getScale)
        .def("setOffset", &Multiply::setOffset)
        .def("getOffset", &Multiply::getOffset)
        .def("setPlanes", &Multiply::setPlanes)
        .def("getPlanes", &Multiply::getPlanes)
        ;
}