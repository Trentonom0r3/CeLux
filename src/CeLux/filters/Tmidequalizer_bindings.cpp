#include "Tmidequalizer_bindings.hpp"

namespace py = pybind11;

void bind_Tmidequalizer(py::module_ &m) {
    py::class_<Tmidequalizer, FilterBase, std::shared_ptr<Tmidequalizer>>(m, "Tmidequalizer")
        .def(py::init<int, float, int>(),
             py::arg("radius") = 5,
             py::arg("sigma") = 0.50,
             py::arg("planes") = 15)
        .def("setRadius", &Tmidequalizer::setRadius)
        .def("getRadius", &Tmidequalizer::getRadius)
        .def("setSigma", &Tmidequalizer::setSigma)
        .def("getSigma", &Tmidequalizer::getSigma)
        .def("setPlanes", &Tmidequalizer::setPlanes)
        .def("getPlanes", &Tmidequalizer::getPlanes)
        ;
}