#include "Dblur_bindings.hpp"

namespace py = pybind11;

void bind_Dblur(py::module_ &m) {
    py::class_<Dblur, FilterBase, std::shared_ptr<Dblur>>(m, "Dblur")
        .def(py::init<float, float, int>(),
             py::arg("angle") = 45.00,
             py::arg("radius") = 5.00,
             py::arg("planes") = 15)
        .def("setAngle", &Dblur::setAngle)
        .def("getAngle", &Dblur::getAngle)
        .def("setRadius", &Dblur::setRadius)
        .def("getRadius", &Dblur::getRadius)
        .def("setPlanes", &Dblur::setPlanes)
        .def("getPlanes", &Dblur::getPlanes)
        ;
}