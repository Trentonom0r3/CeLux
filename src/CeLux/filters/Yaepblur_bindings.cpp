#include "Yaepblur_bindings.hpp"

namespace py = pybind11;

void bind_Yaepblur(py::module_ &m) {
    py::class_<Yaepblur, FilterBase, std::shared_ptr<Yaepblur>>(m, "Yaepblur")
        .def(py::init<int, int, int>(),
             py::arg("radius") = 3,
             py::arg("planes") = 1,
             py::arg("sigma") = 128)
        .def("setRadius", &Yaepblur::setRadius)
        .def("getRadius", &Yaepblur::getRadius)
        .def("setPlanes", &Yaepblur::setPlanes)
        .def("getPlanes", &Yaepblur::getPlanes)
        .def("setSigma", &Yaepblur::setSigma)
        .def("getSigma", &Yaepblur::getSigma)
        ;
}