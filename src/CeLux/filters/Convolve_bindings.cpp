#include "Convolve_bindings.hpp"

namespace py = pybind11;

void bind_Convolve(py::module_ &m) {
    py::class_<Convolve, FilterBase, std::shared_ptr<Convolve>>(m, "Convolve")
        .def(py::init<int, int, float>(),
             py::arg("planes") = 7,
             py::arg("impulse") = 1,
             py::arg("noise") = 0.00)
        .def("setPlanes", &Convolve::setPlanes)
        .def("getPlanes", &Convolve::getPlanes)
        .def("setImpulse", &Convolve::setImpulse)
        .def("getImpulse", &Convolve::getImpulse)
        .def("setNoise", &Convolve::setNoise)
        .def("getNoise", &Convolve::getNoise)
        ;
}