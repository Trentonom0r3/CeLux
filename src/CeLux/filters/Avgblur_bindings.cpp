#include "Avgblur_bindings.hpp"

namespace py = pybind11;

void bind_Avgblur(py::module_ &m) {
    py::class_<Avgblur, FilterBase, std::shared_ptr<Avgblur>>(m, "Avgblur")
        .def(py::init<int, int, int>(),
             py::arg("sizeX") = 1,
             py::arg("planes") = 15,
             py::arg("sizeY") = 0)
        .def("setSizeX", &Avgblur::setSizeX)
        .def("getSizeX", &Avgblur::getSizeX)
        .def("setPlanes", &Avgblur::setPlanes)
        .def("getPlanes", &Avgblur::getPlanes)
        .def("setSizeY", &Avgblur::setSizeY)
        .def("getSizeY", &Avgblur::getSizeY)
        ;
}