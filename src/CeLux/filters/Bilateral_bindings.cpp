#include "Bilateral_bindings.hpp"

namespace py = pybind11;

void bind_Bilateral(py::module_ &m) {
    py::class_<Bilateral, FilterBase, std::shared_ptr<Bilateral>>(m, "Bilateral")
        .def(py::init<float, float, int>(),
             py::arg("sigmaS") = 0.10,
             py::arg("sigmaR") = 0.10,
             py::arg("planes") = 1)
        .def("setSigmaS", &Bilateral::setSigmaS)
        .def("getSigmaS", &Bilateral::getSigmaS)
        .def("setSigmaR", &Bilateral::setSigmaR)
        .def("getSigmaR", &Bilateral::getSigmaR)
        .def("setPlanes", &Bilateral::setPlanes)
        .def("getPlanes", &Bilateral::getPlanes)
        ;
}