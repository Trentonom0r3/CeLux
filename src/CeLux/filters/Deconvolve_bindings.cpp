#include "Deconvolve_bindings.hpp"

namespace py = pybind11;

void bind_Deconvolve(py::module_ &m) {
    py::class_<Deconvolve, FilterBase, std::shared_ptr<Deconvolve>>(m, "Deconvolve")
        .def(py::init<int, int, float>(),
             py::arg("planes") = 7,
             py::arg("impulse") = 1,
             py::arg("noise") = 0.00)
        .def("setPlanes", &Deconvolve::setPlanes)
        .def("getPlanes", &Deconvolve::getPlanes)
        .def("setImpulse", &Deconvolve::setImpulse)
        .def("getImpulse", &Deconvolve::getImpulse)
        .def("setNoise", &Deconvolve::setNoise)
        .def("getNoise", &Deconvolve::getNoise)
        ;
}